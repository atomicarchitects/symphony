"""Generates molecules from a trained model."""

from typing import Sequence, Tuple, Callable, Optional, Union

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
import time

sys.path.append("..")

import analyses.analysis as analysis
from symphony import datatypes
from symphony.data import input_pipeline
from symphony import models

FLAGS = flags.FLAGS


def append_predictions(
    pred: datatypes.Predictions, padded_fragment: datatypes.Fragments, nn_cutoff: float,
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


def generate_one_step(
    padded_fragment: datatypes.Fragments,
    stop: bool,
    rng: chex.PRNGKey,
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    nn_cutoff: float,
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
    init_fragment: datatypes.Fragments,
    max_num_atoms: int,
    cutoff: float,
    rng: chex.PRNGKey,
    return_intermediates: bool = False,
) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
    """Generates a single molecule for a given seed."""
    step_rngs = jax.random.split(rng, num=max_num_atoms)
    (final_padded_fragment, stop), (padded_fragments, preds) = jax.lax.scan(
        lambda args, rng: generate_one_step(*args, rng, apply_fn, cutoff),
        (init_fragment, False),
        step_rngs,
    )
    if return_intermediates:
        return padded_fragments, preds
    else:
        return final_padded_fragment, stop


def generate_molecules(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: str,
    num_seeds: int,
    num_seeds_per_chunk: int,
    init_molecules: Sequence[Union[str, ase.Atoms]],
    max_num_atoms: int,
    visualize: bool,
    steps_for_weight_averaging: Optional[Sequence[int]] = None
):
    """Generates molecules from a trained model at the given workdir."""
    # Check that we can divide the seeds into chunks properly.
    if num_seeds % num_seeds_per_chunk != 0:
        raise ValueError(
            f"num_seeds ({num_seeds}) must be divisible by num_seeds_per_chunk ({num_seeds_per_chunk})"
        )

    # Create initial molecule, if provided.
    if isinstance(init_molecules, str):
        init_molecule, init_molecule_name = analysis.construct_molecule(init_molecules)
        logging.info(
            f"Initial molecule: {init_molecule.get_chemical_formula()} with numbers {init_molecule.numbers} and positions {init_molecule.positions}"
        )
        init_molecules = [init_molecule] * num_seeds
        init_molecule_names = [init_molecule_name] * num_seeds
    else:
        assert len(init_molecules) == num_seeds
        init_molecule_names = [init_molecule.get_chemical_formula() for init_molecule in init_molecules]

    # Load model.
    name = analysis.name_from_workdir(workdir)
    if steps_for_weight_averaging is not None:
        logging.info("Loading model averaged from steps %s", steps_for_weight_averaging)
        model, params, config = analysis.load_weighted_average_model_at_steps(
            workdir, steps_for_weight_averaging, run_in_evaluation_mode=True
        )
    else:
        model, params, config = analysis.load_model_at_step(
            workdir, step, run_in_evaluation_mode=True
        )
    config = config.unlock()
    if "position_updater" in config:
        del config["position_updater"]
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

    # Prepare initial fragments.
    init_fragments = [
        input_pipeline.ase_atoms_to_jraph_graph(
            init_molecule, models.ATOMIC_NUMBERS, config.nn_cutoff
        ) for init_molecule in init_molecules]
    init_fragments = [jraph.pad_with_graphs(
        init_fragment,
        n_node=(max_num_atoms + 1),
        n_edge=(max_num_atoms + 1) ** 2,
        n_graph=2,
    ) for init_fragment in init_fragments]
    init_fragments = jax.tree_map(lambda *err: np.stack(err), *init_fragments)
    init_fragments = jax.vmap(lambda init_fragment: jax.tree_map(jnp.asarray, init_fragment))(init_fragments)
    print(jax.tree_map(jnp.shape, init_fragments))

    @jax.jit
    def chunk_and_apply(
        params: optax.Params, init_fragments: datatypes.Fragments, rngs: chex.PRNGKey
    ) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
        """Chunks the seeds and applies the model sequentially over all chunks."""

        def apply_on_chunk(
            init_fragments_and_rngs: Tuple[datatypes.Fragments, chex.PRNGKey],
        ) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
            """Applies the model on a single chunk."""
            init_fragments, rngs = init_fragments_and_rngs
            assert len(init_fragments.n_node) == len(rngs)

            apply_fn = lambda padded_fragment, rng: model.apply(
                params,
                rng,
                padded_fragment,
                focus_and_atom_type_inverse_temperature,
                position_inverse_temperature,
            )
            generate_for_one_seed_fn = lambda rng, init_fragment: generate_for_one_seed(
                apply_fn,
                init_fragment,
                max_num_atoms,
                config.nn_cutoff,
                rng,
                return_intermediates=visualize,
            )
            return jax.vmap(generate_for_one_seed_fn)(rngs, init_fragments)

        # Chunk the seeds, apply the model, and unchunk the results.
        init_fragments, rngs = jax.tree_map(lambda arr: jnp.reshape(arr, (num_seeds // num_seeds_per_chunk, num_seeds_per_chunk, *arr.shape[1:])), (init_fragments, rngs))
        results = jax.lax.map(apply_on_chunk, (init_fragments, rngs))
        results = jax.tree_map(lambda arr: arr.reshape((-1, *arr.shape[2:])), results)
        return results

    # Generate molecules for all seeds.
    seeds = jnp.arange(num_seeds)
    rngs = jax.vmap(jax.random.PRNGKey)(seeds)

    # Compute compilation time.
    start_time = time.time()
    chunk_and_apply.lower(params, init_fragments, rngs).compile()
    compilation_time = time.time() - start_time
    logging.info("Compilation time: %.2f s", compilation_time)

    # Generate molecules (and intermediate steps, if visualizing).
    if visualize:
        padded_fragments, preds = chunk_and_apply(params, init_fragments, rngs)
    else:
        final_padded_fragments, stops = chunk_and_apply(params, init_fragments, rngs)

    molecule_list = []
    for seed in tqdm.tqdm(seeds, desc="Visualizing molecules"):
        init_fragment = jax.tree_map(lambda x: x[seed], init_fragments)
        init_molecule_name = init_molecule_names[seed]

        if visualize:
            # Get the padded fragment and predictions for this seed.
            padded_fragments_for_seed = jax.tree_map(
                lambda x: x[seed], padded_fragments
            )
            preds_for_seed = jax.tree_map(lambda x: x[seed], preds)

            figs = []
            for step in range(max_num_atoms):
                if step == 0:
                    padded_fragment = init_fragment
                else:
                    padded_fragment = jax.tree_map(
                        lambda x: x[step - 1], padded_fragments_for_seed
                    )
                pred = jax.tree_map(lambda x: x[step], preds_for_seed)

                # Save visualization of generation process.
                fragment = jraph.unpad_with_graphs(padded_fragment)
                pred = jraph.unpad_with_graphs(pred)
                fragment = fragment._replace(
                    globals=jax.tree_map(
                        lambda x: np.squeeze(x, axis=0), fragment.globals
                    )
                )
                pred = pred._replace(
                    globals=jax.tree_map(lambda x: np.squeeze(x, axis=0), pred.globals)
                )
                fig = analysis.visualize_predictions(pred, fragment)
                figs.append(fig)

                # This may be the final padded fragment.
                final_padded_fragment = padded_fragment

                # Check if we should stop.
                stop = pred.globals.stop
                if stop:
                    break

            # Save the visualizations of the generation process.
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

        else:
            # We already have the final padded fragment.
            final_padded_fragment = jax.tree_map(
                lambda x: x[seed], final_padded_fragments
            )
            stop = jax.tree_map(lambda x: x[seed], stops)

        num_valid_nodes = final_padded_fragment.n_node[0]
        generated_molecule = ase.Atoms(
            positions=final_padded_fragment.nodes.positions[:num_valid_nodes],
            numbers=models.get_atomic_numbers(
                final_padded_fragment.nodes.species[:num_valid_nodes]
            ),
        )
        if stop:
            logging.info("Generated %s", generated_molecule.get_chemical_formula())
            outputfile = f"{init_molecule_name}_seed={seed}.xyz"
        else:
            logging.info("STOP was not produced. Discarding...")
            outputfile = f"{init_molecule_name}_seed={seed}_no_stop.xyz"

        ase.io.write(os.path.join(molecules_outputdir, outputfile), generated_molecule)
        molecule_list.append(generated_molecule)

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
    init_molecule = FLAGS.init
    max_num_atoms = FLAGS.max_num_atoms
    visualize = FLAGS.visualize
    steps_for_weight_averaging = FLAGS.steps_for_weight_averaging

    generate_molecules(
        workdir,
        outputdir,
        focus_and_atom_type_inverse_temperature,
        position_inverse_temperature,
        step,
        num_seeds,
        num_seeds_per_chunk,
        init_molecule,
        max_num_atoms,
        visualize,
        steps_for_weight_averaging
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
    flags.DEFINE_list(
        "steps_for_weight_averaging",
        None,
        "Steps to average parameters over. If None, the model at the given step is used.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
