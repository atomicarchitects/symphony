"""Visualize the fragments and corresponding predictions."""

from typing import Sequence
import os

from absl import flags
from absl import app
from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import jraph

from symphony.data import input_pipeline

from analyses import analysis
from analyses import generate_molecules_multifocus
from analyses import visualizer

FLAGS = flags.FLAGS


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    visualize_predictions_and_fragments(
        workdir,
        FLAGS.outputdir,
        FLAGS.focus_and_atom_type_inverse_temperature,
        FLAGS.position_inverse_temperature,
        FLAGS.step,
        FLAGS.init,
        FLAGS.max_num_atoms,
        FLAGS.merge_cutoff,
        FLAGS.seed,
    )


def visualize_predictions_and_fragments(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: str,
    init_molecule: str,
    max_num_atoms: int,
    merge_cutoff: float,
    seed: int,
):
    """Visualize the predictions and fragments."""
    # Create initial molecule, if provided.
    init_molecule, init_molecule_name = analysis.construct_molecule(init_molecule)
    logging.info(
        f"Initial molecule: {init_molecule.get_chemical_formula()} with numbers {init_molecule.numbers} and positions {init_molecule.positions}"
    )

    # Load the model.
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )

    # Create the initial fragment.
    init_fragment = input_pipeline.ase_atoms_to_jraph_graph(
        init_molecule,
        atomic_numbers=np.asarray([1, 6, 7, 8, 9]),
        nn_cutoff=config.nn_cutoff,
    )

    # Load the dataset.
    rng = jax.random.PRNGKey(seed)

    all_fragments = []
    all_preds = []
    fragment = init_fragment
    while fragment.n_node.sum() < max_num_atoms:
        # Pad the fragment.
        fragment = jraph.pad_with_graphs(fragment, n_node=max_num_atoms, n_edge=max_num_atoms * max_num_atoms, n_graph=2)
        
        apply_rng, rng = jax.random.split(rng)
        preds = jax.jit(model.apply)(
            params,
            apply_rng,
            fragment,
            focus_and_atom_type_inverse_temperature,
            position_inverse_temperature,
        )

        # Remove padding graphs.
        fragment = jraph.unpad_with_graphs(fragment)
        preds = jraph.unpad_with_graphs(preds)

        # Save the fragment and predictions.
        all_fragments.append(fragment)
        all_preds.append(preds)

        # Update the fragment.
        old_fragment = fragment
        stop, fragment = generate_molecules_multifocus.append_predictions_to_fragment(
            fragment, preds, merge_cutoff, config.nn_cutoff
        )
        if stop:
            break

        # Print the distance matrix.
        print(f"Distance matrix for fragment:")
        print(jnp.linalg.norm(fragment.nodes.positions[:, None, :] - fragment.nodes.positions[None, :, :], axis=-1))

        if np.all(fragment.n_node == old_fragment.n_node):
            break

    # We create one figure per fragment.
    figs = []
    for fragment, pred in zip(all_fragments, all_preds):
        figs.append(visualizer.visualize_predictions(pred, fragment))

    # Save to files.
    visualizations_dir = os.path.join(
        outputdir,
        name,
        f"fait={focus_and_atom_type_inverse_temperature}",
        f"pit={position_inverse_temperature}",
        f"step={step}",
        "visualizations",
        "generated_fragments",
    )
    os.makedirs(
        visualizations_dir,
        exist_ok=True,
    )
    for index in range(len(figs)):
        outputfile = os.path.join(
            visualizations_dir,
            f"{init_molecule_name}_seed={seed}_fragments_{index}.html",
        )
        figs[index].write_html(outputfile, include_plotlyjs="cdn")


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
    flags.DEFINE_float(
        "merge_cutoff",
        0.5,
        "Cutoff for merging atoms.",
    )
    flags.DEFINE_integer(
        "seed",
        0,
        "Seed for the random number generator.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
