"""Generates molecules from a trained model."""

from typing import Sequence, Tuple, Callable

import os
import sys

from absl import flags
from absl import app
from absl import logging
import ase
import ase.data
from ase.db import connect
import ase.io
import ase.visualize
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import tqdm
import chex
import optax

sys.path.append("..")

import analyses.analysis as analysis
from symphony import datatypes
from symphony.data import input_pipeline
from symphony.models import models

FLAGS = flags.FLAGS


def append_predictions(
    pred: datatypes.Predictions, padded_fragment: datatypes.Fragments, nn_cutoff: float
) -> datatypes.Fragments:
    """Appends the predictions to the padded fragment."""
    # Update the positions of the first dummy node.
    positions = padded_fragment.nodes.positions
    num_valid_nodes = padded_fragment.n_node[0]
    num_nodes = padded_fragment.nodes.positions.shape[0]
    focus = pred.globals.focus_indices[0]
    focus_position = positions[focus]
    target_position = pred.globals.position_vectors[0] + focus_position
    new_positions = positions.at[num_valid_nodes].set(target_position)

    # Update the species of the first dummy node.
    species = padded_fragment.nodes.species
    target_species = pred.globals.target_species[0]
    new_species = species.at[num_valid_nodes].set(target_species)

    # Compute the distance matrix to select the edges.
    distance_matrix = jnp.linalg.norm(
        new_positions[None, :, :] - new_positions[:, None, :], axis=-1
    )
    node_indices = jnp.arange(num_nodes)

    # Avoid self-edges.
    valid_edges = (distance_matrix > 0) & (distance_matrix < nn_cutoff)
    valid_edges = (
        valid_edges
        & (node_indices[None, :] <= num_valid_nodes)
        & (node_indices[:, None] <= num_valid_nodes)
    )
    senders, receivers = jnp.nonzero(
        valid_edges, size=num_nodes * num_nodes, fill_value=-1
    )
    num_valid_edges = jnp.sum(valid_edges)
    num_valid_nodes += 1

    return padded_fragment._replace(
        nodes=padded_fragment.nodes._replace(
            positions=new_positions,
            species=new_species,
        ),
        n_node=jnp.asarray([num_valid_nodes, num_nodes - num_valid_nodes]),
        n_edge=jnp.asarray([num_valid_edges, num_nodes * num_nodes - num_valid_edges]),
        senders=senders,
        receivers=receivers,
    )


# Generate with different seeds.
def generate_one_step(
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    padded_fragment: datatypes.Fragments,
    stop: bool,
    nn_cutoff: float,
    rng: chex.PRNGKey,
) -> Tuple[
    Tuple[datatypes.Fragments, bool], Tuple[datatypes.Fragments, datatypes.Predictions]
]:
    """Generates the next fragment for a given seed."""
    pred = apply_fn(padded_fragment, rng)
    next_padded_fragment = append_predictions(pred, padded_fragment, nn_cutoff)
    stop = pred.globals.stop[0] | stop
    return jax.lax.cond(
        stop,
        lambda: ((padded_fragment, True), (padded_fragment, pred)),
        lambda: ((next_padded_fragment, False), (next_padded_fragment, pred)),
    )


def generate_for_one_seed(
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    init_fragment,
    max_num_atoms,
    cutoff: float,
    rng: chex.PRNGKey,
) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
    """Generates a single molecule for a given seed."""
    step_rngs = jax.random.split(rng, num=max_num_atoms)
    _, (padded_fragments, preds) = jax.lax.scan(
        lambda args, rng: generate_one_step(apply_fn, *args, cutoff, rng),
        (init_fragment, False),
        step_rngs,
    )
    return padded_fragments, preds


def generate_molecules(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: str,
    num_seeds: int,
    num_seeds_per_chunk: int,
    init_molecule: str,
    max_num_atoms: int,
    visualize: bool,
):
    """Generates molecules from a trained model at the given workdir."""
    # Create initial molecule, if provided.
    init_molecule, init_molecule_name = analysis.construct_molecule(init_molecule)
    logging.info(
        f"Initial molecule: {init_molecule.get_chemical_formula()} with numbers {init_molecule.numbers} and positions {init_molecule.positions}"
    )

    # Load model.
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )
    logging.info(config.to_dict())

    # Create output directories.
    molecules_outputdir = os.path.join(
        outputdir,
        name,
        f"fait={focus_and_atom_type_inverse_temperature}",
        f"pit={position_inverse_temperature}",
        f"step={step}",
        "molecules",
    )
    os.makedirs(molecules_outputdir, exist_ok=True)
    if visualize:
        visualizations_dir = os.path.join(
            outputdir,
            name,
            f"fait={focus_and_atom_type_inverse_temperature}",
            f"pit={position_inverse_temperature}",
            f"step={step}",
            "visualizations",
            "generated_molecules",
        )
        os.makedirs(visualizations_dir, exist_ok=True)
    molecule_list = []

    # Prepare initial fragment.
    init_fragment = input_pipeline.ase_atoms_to_jraph_graph(
        init_molecule, models.ATOMIC_NUMBERS, config.nn_cutoff
    )
    init_fragment = jraph.pad_with_graphs(
        init_fragment,
        n_node=(max_num_atoms + 1),
        n_edge=(max_num_atoms + 1) ** 2,
        n_graph=2,
    )
    init_fragment = jax.tree_map(jnp.asarray, init_fragment)

    # Generate molecules, for all seeds
    def apply_with_params(
        params: optax.Params, rngs: chex.PRNGKey
    ) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
        """Helper to avoid folding-in params."""
        apply_fn = lambda padded_fragment, rng: model.apply(
            params,
            rng,
            padded_fragment,
            focus_and_atom_type_inverse_temperature,
            position_inverse_temperature,
        )
        generate_for_one_seed_fn = lambda rng: generate_for_one_seed(
            apply_fn, init_fragment, max_num_atoms, config.nn_cutoff, rng
        )
        vmapped_generate_fn = jax.vmap(generate_for_one_seed_fn)
        padded_fragments, preds = vmapped_generate_fn(rngs)
        return padded_fragments, preds

    def apply(params, rngs):
        return jax.lax.map(lambda chunk_rngs: apply_with_params(params, chunk_rngs), rngs)

    # Generate molecules for all seeds.
    seeds = jnp.arange(num_seeds)
    rngs = jax.vmap(jax.random.PRNGKey)(seeds)
    rngs = rngs.reshape((num_seeds // num_seeds_per_chunk, num_seeds_per_chunk, -1))
    padded_fragments, preds = apply(params, rngs)
    padded_fragments, preds = jax.tree_map(lambda x: x.reshape((-1, *x.shape[2:])), (padded_fragments, preds))

    for seed in tqdm.tqdm(seeds, desc="Visualizing molecules"):
        # Get the padded fragment and predictions for this seed.
        padded_fragments_for_seed = jax.tree_map(lambda x: x[seed], padded_fragments)
        preds_for_seed = jax.tree_map(lambda x: x[seed], preds)

        figs = []
        nan_found = False
        final_fragment = None
        for step in range(max_num_atoms):
            padded_fragment = jax.tree_map(lambda x: x[step], padded_fragments_for_seed)
            pred = jax.tree_map(lambda x: x[step], preds_for_seed)

            # Check for any NaNs in the predictions.
            num_nans = sum(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(lambda x: jnp.isnan(x).sum(), pred)
                )
            )
            if num_nans > 0:
                logging.info(
                    "NaNs in predictions at step %d. Stopping generation...", step
                )
                nan_found = True
                final_fragment = jax.tree_map(
                    lambda x: x[step - 1], padded_fragments_for_seed
                )
                break

            # Save visualization of generation process.
            if visualize:
                fragment = jraph.unpad_with_graphs(padded_fragment)
                fig = analysis.visualize_predictions(pred, fragment)
                figs.append(fig)

            # Check if we should stop.
            stop = pred.globals.stop[0]
            if stop:
                final_fragment = padded_fragment
                break

        # We don't generate molecules with more than MAX_NUM_ATOMS atoms.
        if final_fragment is None:
            logging.info("No final fragment found. Skipping...")
            continue

        num_valid_nodes = final_fragment.n_node[0]
        generated_molecule = ase.Atoms(
            positions=final_fragment.nodes.positions[:num_valid_nodes],
            numbers=models.get_atomic_numbers(
                final_fragment.nodes.species[:num_valid_nodes]
            ),
        )
        logging.info("Generated %s", generated_molecule.get_chemical_formula())
        if nan_found:
            outputfile = f"{init_molecule_name}_seed={seed}_NaN.xyz"
        else:
            outputfile = f"{init_molecule_name}_seed={seed}.xyz"
        ase.io.write(os.path.join(molecules_outputdir, outputfile), generated_molecule)
        molecule_list.append(generated_molecule)

        if visualize:
            for index, fig in enumerate(figs):
                # Update the title.
                model_name = analysis.get_title_for_name(name)
                fig.update_layout(
                    title=f"{model_name}: Predictions for Seed {seed}",
                    title_x=0.5,
                )

                # Save to file.
                outputfile = os.path.join(
                    visualizations_dir,
                    f"seed_{seed}_fragments_{index}.html",
                )
                fig.write_html(outputfile, include_plotlyjs="cdn")

    # Save the generated molecules as an ASE database.
    output_db = os.path.join(
        molecules_outputdir, f"generated_molecules_init={init_molecule_name}.db"
    )
    with connect(output_db) as conn:
        for mol in molecule_list:
            conn.write(mol)


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    outputdir = FLAGS.outputdir
    focus_and_atom_type_inverse_temperature = (
        FLAGS.focus_and_atom_type_inverse_temperature
    )
    position_inverse_temperature = FLAGS.position_inverse_temperature
    step = FLAGS.step
    num_seeds = FLAGS.num_seeds
    num_seeds_per_chunk = FLAGS.num_seeds_per_chunk
    init = FLAGS.init
    max_num_atoms = FLAGS.max_num_atoms
    visualize = FLAGS.visualize

    generate_molecules(
        workdir,
        outputdir,
        focus_and_atom_type_inverse_temperature,
        position_inverse_temperature,
        step,
        num_seeds,
        num_seeds_per_chunk,
        init,
        max_num_atoms,
        visualize,
    )


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for model.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses", "analysed_workdirs"),
        "Directory where molecules should be saved.",
    )
    flags.DEFINE_float(
        "focus_and_atom_type_inverse_temperature",
        1.0,
        "Inverse temperature value for sampling the focus and atom type.",
        short_name="fait",
    )
    flags.DEFINE_float(
        "position_inverse_temperature",
        1.0,
        "Inverse temperature value for sampling the position.",
        short_name="pit",
    )
    flags.DEFINE_string(
        "step",
        "best",
        "Step number to load model from. The default corresponds to the best model.",
    )
    flags.DEFINE_integer(
        "num_seeds",
        128,
        "Seeds to attempt to generate molecules from.",
    )
    flags.DEFINE_integer(
        "num_seeds_per_chunk",
        32,
        "Number of seeds evaluated in parallel. Reduce to avoid OOM errors.",
    )
    flags.DEFINE_string(
        "init",
        "C",
        "An initial molecular fragment to start the generation process from.",
    )
    flags.DEFINE_integer(
        "max_num_atoms",
        30,
        "Maximum number of atoms to generate per molecule.",
    )
    flags.DEFINE_bool(
        "visualize",
        False,
        "Whether to visualize the generation process step-by-step.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
