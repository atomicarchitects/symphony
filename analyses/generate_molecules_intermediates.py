"""Generates molecules from a trained model."""

from typing import Sequence, Tuple, Callable, Optional, Union

import os

from absl import flags
from absl import app
from absl import logging
import ase
import ase.data
from ase.db import connect
import ase.io
import ase.visualize
import itertools
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import tqdm
import chex
import optax
import time

import analyses.analysis as analysis
import analyses.visualizer as visualizer
from symphony import datatypes
from symphony.data import input_pipeline
from symphony import models

FLAGS = flags.FLAGS
AVG_EDGES_PER_NODE = 10


def append_predictions(
    pred: datatypes.Predictions,
    padded_fragment: datatypes.Fragments,
    nn_cutoff: float,
) -> datatypes.Fragments:
    """Appends the predictions to the padded fragment."""
    # Update the positions of the first dummy node.
    positions = jnp.asarray(padded_fragment.nodes.positions)
    num_valid_nodes = padded_fragment.n_node[0]
    num_nodes = padded_fragment.nodes.positions.shape[0]
    focus = pred.globals.focus_indices[0]
    focus_position = positions[focus]
    target_position = pred.globals.position_vectors[0] + focus_position
    new_positions = positions.at[num_valid_nodes].set(target_position)

    # Update the species of the first dummy node.
    species = jnp.asarray(padded_fragment.nodes.species)
    target_species = pred.globals.target_species[0]
    new_species = species.at[num_valid_nodes].set(target_species)

    # Compute the distance matrix to select the edges.
    distance_matrix = jnp.linalg.norm(
        new_positions[None, :, :] - new_positions[:, None, :], axis=-1
    )
    node_indices = jnp.arange(num_nodes)

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
        n_edge=jnp.asarray([num_valid_edges, num_nodes * AVG_EDGES_PER_NODE - num_valid_edges]),
        senders=senders,
        receivers=receivers,
    )


def generate_molecules(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: str,
    num_seeds: int,
    init_molecules: Sequence[
        Union[str, ase.Atoms, jraph.GraphsTuple, datatypes.Fragments]
    ],
    max_num_atoms: int,
    max_steps: int,
    steps_for_weight_averaging: Optional[Sequence[int]] = None,
    filetype: str = "xyz",
    visualize: bool = False,
    res_alpha: Optional[int] = None,
    res_beta: Optional[int] = None,
):
    """Generates molecules from a trained model at the given workdir."""

    # Load model.
    name = analysis.name_from_workdir(workdir)
    if steps_for_weight_averaging is not None:
        logging.info("Loading model averaged from steps %s", steps_for_weight_averaging)
        model, params, config = analysis.load_weighted_average_model_at_steps(
            workdir, steps_for_weight_averaging, run_in_evaluation_mode=True
        )
    else:
        model, params, config = analysis.load_model_at_step(
            workdir,
            step,
            run_in_evaluation_mode=True,
            res_alpha=res_alpha,
            res_beta=res_beta,
        )
    config = config.unlock()
    logging.info(config.to_dict())

    # Create initial molecule, if provided.
    if isinstance(init_molecules, str):
        init_molecule, init_molecule_name = analysis.construct_molecule(init_molecules)
        logging.info(
            f"Initial molecule: {init_molecule.get_chemical_formula()} with numbers {init_molecule.numbers} and positions {init_molecule.positions}"
        )
        init_molecules = [init_molecule] * num_seeds
        init_molecule_names = [init_molecule_name] * num_seeds
        init_molecules = [
            input_pipeline.ase_atoms_to_jraph_graph(
                init_molecule,
                models.ATOMIC_NUMBERS,
                config.nn_cutoff,
            )
            for init_molecule in init_molecules
        ]
    elif isinstance(init_molecules[0], ase.Atoms):
        assert len(init_molecules) == num_seeds
        init_molecule_names = [
            f"mol_{i}_{init_molecule.get_chemical_formula()}"
            for i, init_molecule in enumerate(init_molecules)
        ]
        init_molecules = [
            input_pipeline.ase_atoms_to_jraph_graph(
                init_molecule,
                models.ATOMIC_NUMBERS,
                config.nn_cutoff,
            )
            for init_molecule in init_molecules
        ]
    elif isinstance(init_molecules[0], jraph.GraphsTuple) or isinstance(
        init_molecules[0], datatypes.Fragments
    ):
        init_molecule_names = [f"mol_{i}" for i in range(len(init_molecules))]
    else:
        raise TypeError(
            "input molecules must be a list of strings, ASE Atoms, or jraph.GraphsTuples"
        )

    apply_fn = lambda padded_fragment, rng: model.apply(
        params,
        rng,
        padded_fragment,
        focus_and_atom_type_inverse_temperature,
        position_inverse_temperature,
    )

    # Create output directories.
    molecules_outputdir = os.path.join(
        outputdir,
        name,
        f"fait={focus_and_atom_type_inverse_temperature}",
        f"pit={position_inverse_temperature}",
        f"step={step}",
    )
    if res_alpha is not None:
        molecules_outputdir += f"_res_alpha={res_alpha}"
    if res_beta is not None:
        molecules_outputdir += f"_res_beta={res_beta}"
    molecules_outputdir += "/molecules"
    print(f"Creating output directory {molecules_outputdir}")
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

    init_fragments = [
        jraph.pad_with_graphs(
            init_fragment,
            n_node=(max_num_atoms + 1),
            n_edge=(max_num_atoms + 1) * AVG_EDGES_PER_NODE,
            n_graph=2,
        )
        for init_fragment in init_molecules
    ]
    init_fragments = jax.tree_map(lambda *err: np.stack(err), *init_fragments)
    init_fragments = jax.vmap(
        lambda init_fragment: jax.tree_map(jnp.asarray, init_fragment)
    )(init_fragments)

    def generator(
        frag: datatypes.Fragments, rng_gen: chex.PRNGKey, max_steps: int
    ) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
        step_rngs = jax.random.split(rng_gen, num=max_steps)
        stop = False
        current_frag = frag
        pred = apply_fn(current_frag, step_rngs[0])
        for step_rng in step_rngs:
            pred = apply_fn(current_frag, step_rng)
            next_padded_fragment = append_predictions(pred, current_frag, config.nn_cutoff)
            stop = pred.globals.stop[0] or stop
            if stop:
                yield current_frag, pred, stop
            else:
                yield next_padded_fragment, pred, stop
            current_frag = next_padded_fragment  # this is where n_edge padding goes wild

    # Generate molecules for all seeds.
    seeds = jnp.arange(num_seeds)
    rngs = jax.vmap(jax.random.PRNGKey)(seeds)

    # Compute compilation time.
    start_time = time.time()
    # chunk_and_apply.lower(params, init_fragments, rngs).compile()
    compilation_time = time.time() - start_time
    logging.info("Compilation time: %.2f s", compilation_time)

    # Generate molecules (and intermediate steps, if visualizing).
    molecule_list = []
    for seed in tqdm.tqdm(seeds, desc="Visualizing molecules"):
        rng = rngs[seed]
        init_molecule_name = init_molecule_names[seed]
        init_fragment = jax.tree_map(lambda x: x[seed], init_fragments)

        step = 0
        figs = []
        final_padded_fragment = init_fragment
        for padded_fragment, pred, stop in generator(init_fragment, rng, max_steps):
            # print(padded_fragment.n_node, jraph.get_number_of_padding_with_graphs_graphs(padded_fragment))
            fragment = jraph.unpad_with_graphs(padded_fragment)
            fragment = fragment._replace(
                globals=jax.tree_map(
                    lambda x: np.squeeze(x, axis=0) if x.shape[0] == 1 else x, fragment.globals
                )
            )
            pred = jraph.unpad_with_graphs(pred)
            pred = pred._replace(
                globals=jax.tree_map(lambda x: np.squeeze(x, axis=0) if x.shape[0] == 1 else x, pred.globals)
            )
            if visualize:
                fig = visualizer.visualize_predictions(pred, fragment)
                figs.append(fig)

            # This may be the final padded fragment.
            final_padded_fragment = padded_fragment

            # Check if we should stop.
            stop = pred.globals.stop
            if stop:
                break
            if fragment.n_node[0] >= max_num_atoms:
                break

        # Save the visualizations of the generation process.
        if visualize:
            for index, fig in enumerate(figs):
                # Update the title.
                model_name = visualizer.get_title_for_name(name)
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

        num_valid_nodes = final_padded_fragment.n_node[0]
        generated_molecule = ase.Atoms(
            positions=final_padded_fragment.nodes.positions[:num_valid_nodes],
            numbers=models.get_atomic_numbers(
                final_padded_fragment.nodes.species[:num_valid_nodes]
            ),
        )
        if stop:
            logging.info("Generated %s", generated_molecule.get_chemical_formula())
            print("Generated %s", generated_molecule.get_chemical_formula())
            outputfile = f"{init_molecule_name}_seed={seed}.{filetype}"
        else:
            logging.info("STOP was not produced. Discarding...")
            print("STOP was not produced. Discarding...")
            outputfile = f"{init_molecule_name}_seed={seed}_no_stop.{filetype}"

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
    init_molecule = FLAGS.init
    max_num_atoms = FLAGS.max_num_atoms
    max_num_steps = FLAGS.max_num_steps
    steps_for_weight_averaging = FLAGS.steps_for_weight_averaging
    filetype = FLAGS.filetype
    visualize = FLAGS.visualize

    generate_molecules(
        workdir,
        outputdir,
        focus_and_atom_type_inverse_temperature,
        position_inverse_temperature,
        step,
        num_seeds,
        init_molecule,
        max_num_atoms,
        max_num_steps,
        steps_for_weight_averaging,
        filetype,
        visualize,
        res_alpha=FLAGS.res_alpha,
        res_beta=FLAGS.res_beta,
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
        "res_alpha",
        None,
        "Angular resolution of alpha.",
    )
    flags.DEFINE_integer(
        "res_beta",
        None,
        "Angular resolution of beta.",
    )
    flags.DEFINE_integer(
        "num_seeds",
        128,
        "Seeds to attempt to generate molecules from.",
    )
    flags.DEFINE_string(
        "init",
        "O",
        "An initial molecular fragment to start the generation process from.",
    )
    flags.DEFINE_integer(
        "max_num_atoms",
        100,
        "Maximum number of atoms in molecule.",
    )
    flags.DEFINE_integer(
        "max_num_steps",
        100,
        "Maximum number of atoms to add.",
    )
    flags.DEFINE_list(
        "steps_for_weight_averaging",
        None,
        "Steps to average parameters over. If None, the model at the given step is used.",
    )
    flags.DEFINE_string(
        "filetype",
        "xyz",
        "File extension to use (xyz or cif).",
    )
    flags.DEFINE_bool(
        "visualize",
        False,
        "Whether to produce visualizations at each step.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
