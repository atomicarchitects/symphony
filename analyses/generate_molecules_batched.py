"""Generates molecules from a trained model."""

import os
import sys
import time
from typing import Callable, Sequence, Tuple

import ase
import ase.data
import ase.io
import ase.visualize
import chex
import jax
import jax.numpy as jnp
import jraph
import optax
import tqdm
from absl import app, flags, logging
from ase.db import connect

sys.path.append("..")

import analyses.analysis as analysis
from symphony import datatypes
from symphony.data import input_pipeline
from symphony import models

FLAGS = flags.FLAGS


def append_predictions(
    preds: datatypes.Predictions,
    stop: chex.Array,
    padded_fragments: datatypes.Fragments,
    radial_cutoff: float,
) -> datatypes.Fragments:
    """Appends the predictions to the padded fragment."""
    num_nodes = padded_fragments.nodes.positions.shape[0]
    num_graphs = padded_fragments.n_node.shape[0]
    num_padding_nodes = padded_fragments.n_node[-1]
    num_valid_nodes = num_nodes - num_padding_nodes
    num_padding_graphs = 1
    num_valid_graphs = num_graphs - num_padding_graphs
    num_edges = padded_fragments.senders.shape[0]

    #
    num_unstopped_graphs = (~stop).sum()
    dummy_nodes_indices = num_valid_nodes + jnp.arange(num_valid_graphs)

    # Update segment ids of the first dummy nodes.
    segment_ids = models.get_segment_ids(padded_fragments.n_node, num_nodes)
    dummy_new_segment_ids = segment_ids[dummy_nodes_indices]
    dummy_new_segment_ids = jnp.where(
        stop, dummy_new_segment_ids, jnp.arange(num_valid_graphs)
    )
    segment_ids = segment_ids.at[dummy_nodes_indices].set(dummy_new_segment_ids)

    # Update positions of the first dummy nodes.
    positions = padded_fragments.nodes.positions
    focuses = preds.globals.focus_indices[:num_valid_graphs]
    focus_positions = positions[focuses]
    target_positions_relative_to_focus = preds.globals.position_vectors[
        :num_valid_graphs
    ]
    target_positions = target_positions_relative_to_focus + focus_positions
    dummy_positions = positions[dummy_nodes_indices]
    dummy_new_positions = jnp.where(stop[:, None], dummy_positions, target_positions)
    positions = positions.at[dummy_nodes_indices].set(dummy_new_positions)

    # Update the species of the first dummy nodes.
    species = padded_fragments.nodes.species
    target_species = preds.globals.target_species[:num_valid_graphs]
    dummy_species = species[dummy_nodes_indices]
    dummy_new_species = jnp.where(stop, dummy_species, target_species)
    species = species.at[dummy_nodes_indices].set(dummy_new_species)

    # Sort nodes according to segment ids.
    sort_indices = jnp.argsort(segment_ids, kind="stable")
    segment_ids = segment_ids[sort_indices]
    positions = positions[sort_indices]
    species = species[sort_indices]

    # Compute the distance matrix to select the edges.
    distance_matrix = jnp.linalg.norm(
        positions[None, :, :] - positions[:, None, :], axis=-1
    )
    node_indices = jnp.arange(num_nodes)

    # Avoid self-edges and linking across graphs.
    valid_edges = (distance_matrix > 0) & (distance_matrix <= radial_cutoff)
    valid_edges = valid_edges & (segment_ids[None, :] == segment_ids[:, None])
    valid_edges = (
        valid_edges
        & (node_indices[None, :] < num_valid_nodes + num_unstopped_graphs)
        & (node_indices[:, None] < num_valid_nodes + num_unstopped_graphs)
    )
    senders, receivers = jnp.nonzero(valid_edges, size=num_edges, fill_value=-1)

    # Update the number of nodes and edges.
    n_node = jnp.bincount(segment_ids, length=num_graphs)
    n_edge = jnp.bincount(segment_ids[senders], length=num_graphs)

    return padded_fragments._replace(
        nodes=padded_fragments.nodes._replace(
            positions=positions,
            species=species,
        ),
        n_node=n_node,
        n_edge=n_edge,
        senders=senders,
        receivers=receivers,
    )


def generate_one_step(
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    padded_fragments: datatypes.Fragments,
    stop: bool,
    radial_cutoff: float,
    rng: chex.PRNGKey,
) -> Tuple[
    Tuple[datatypes.Fragments, bool], Tuple[datatypes.Fragments, datatypes.Predictions]
]:
    """Generates the next fragment for a given seed."""
    preds = apply_fn(padded_fragments, rng)
    stop = preds.globals.stop[:-1] | stop
    next_padded_fragments = append_predictions(preds, stop, padded_fragments, radial_cutoff)
    return (next_padded_fragments, stop), (next_padded_fragments, preds)


def generate_for_one_seed(
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    init_fragments,
    max_num_atoms,
    cutoff: float,
    rng: chex.PRNGKey,
    return_intermediates: bool = False,
) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
    """Generates a single molecule for a given seed."""
    step_rngs = jax.random.split(rng, num=max_num_atoms)
    num_valid_graphs = init_fragments.n_node.shape[0] - 1
    (final_padded_fragments, stop), (padded_fragments, preds) = jax.lax.scan(
        lambda args, rng: generate_one_step(apply_fn, *args, cutoff, rng),
        (init_fragments, jnp.zeros(num_valid_graphs, dtype=bool)),
        step_rngs,
    )
    if return_intermediates:
        return padded_fragments, preds
    else:
        return final_padded_fragments, stop


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
    # Check that we can divide the seeds into chunks properly.
    if num_seeds % num_seeds_per_chunk != 0:
        raise ValueError(
            f"num_seeds ({num_seeds}) must be divisible by num_seeds_per_chunk ({num_seeds_per_chunk})"
        )

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

    # Prepare initial fragment.
    init_fragment = input_pipeline.ase_atoms_to_jraph_graph(
        init_molecule, models.ATOMIC_NUMBERS, config.radial_cutoff
    )
    init_fragments = jraph.batch([init_fragment] * num_seeds_per_chunk)
    init_fragments = jraph.pad_with_graphs(
        init_fragments,
        n_node=(max_num_atoms * num_seeds_per_chunk),
        n_edge=(max_num_atoms * num_seeds_per_chunk * 10),
        n_graph=num_seeds_per_chunk + 1,
    )
    init_fragments = jax.tree_util.tree_map(jnp.asarray, init_fragments)
    (jax.tree_util.tree_map(jnp.shape, init_fragments))

    @jax.jit
    def chunk_and_apply(
        params: optax.Params, rngs: chex.PRNGKey
    ) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
        """Chunks the seeds and applies the model sequentially over all chunks."""

        def apply_on_chunk(
            rng: chex.PRNGKey,
        ) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
            """Applies the model on a single chunk."""
            apply_fn = lambda padded_fragments, rng: model.apply(
                params,
                rng,
                padded_fragments,
                focus_and_atom_type_inverse_temperature,
                position_inverse_temperature,
            )
            return generate_for_one_seed(
                apply_fn,
                init_fragments,
                max_num_atoms,
                config.radial_cutoff,
                rng,
                return_intermediates=visualize,
            )

        return jax.lax.map(apply_on_chunk, rngs)

    # Generate molecules for all seeds.
    seeds = jnp.arange(num_seeds // num_seeds_per_chunk)
    rngs = jax.vmap(jax.random.PRNGKey)(seeds)

    # Compute compilation time.
    start_time = time.time()
    chunk_and_apply.lower(params, rngs).compile()
    compilation_time = time.time() - start_time
    logging.info("Compilation time: %.2f s", compilation_time)

    # Generate molecules (and intermediate steps, if visualizing).
    if visualize:
        padded_fragments, preds = chunk_and_apply(params, rngs)
    else:
        final_padded_fragments, stops = chunk_and_apply(params, rngs)

    molecule_list = []
    for seed in tqdm.tqdm(seeds, desc="Visualizing molecules"):
        final_padded_fragments_for_seed = jax.tree_util.tree_map(
            lambda x: x[seed], final_padded_fragments
        )
        stops_for_seed = jax.tree_util.tree_map(lambda x: x[seed], stops)

        for index, final_padded_fragment in enumerate(
            jraph.unbatch(jraph.unpad_with_graphs(final_padded_fragments_for_seed))
        ):
            generated_molecule = ase.Atoms(
                positions=final_padded_fragment.nodes.positions,
                numbers=models.get_atomic_numbers(final_padded_fragment.nodes.species),
            )

            if stops_for_seed[index]:
                logging.info("Generated %s", generated_molecule.get_chemical_formula())
                outputfile = f"{init_molecule_name}_seed={seed}.xyz"
            else:
                logging.info("STOP was not produced. Discarding...")
                outputfile = f"{init_molecule_name}_seed={seed}_no_stop.xyz"

            ase.io.write(
                os.path.join(molecules_outputdir, outputfile), generated_molecule
            )
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
