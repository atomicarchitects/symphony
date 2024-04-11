"""Creates a series of visualizations when building up a molecule."""

import os
import sys
from typing import Optional, Sequence, List, Tuple

import ase
import ase.build
import ase.data
import ase.io
import ase.visualize
import jax
import jraph
import numpy as np
import tqdm
from absl import app, flags, logging

sys.path.append("..")

from analyses import analysis
from analyses import visualizer
from symphony.data import input_pipeline
from symphony import models

FLAGS = flags.FLAGS


def _remove_target_atoms(
    molecule: ase.Atoms, cutoff: float
) -> Tuple[List[ase.Atoms], List[jraph.GraphsTuple]]:
    """Removes each atom in the molecule and returns the resulting fragments."""
    molecules_with_target_removed = []
    fragments = []
    for target in range(len(molecule)):
        molecule_with_target_removed = ase.Atoms(
            positions=np.concatenate(
                [molecule.positions[:target], molecule.positions[target + 1 :]]
            ),
            numbers=np.concatenate(
                [molecule.numbers[:target], molecule.numbers[target + 1 :]]
            ),
        )
        fragment = input_pipeline.ase_atoms_to_jraph_graph(
            molecule_with_target_removed,
            ATOMIC_NUMBERS,
            cutoff,
        )

        molecules_with_target_removed.append(molecule_with_target_removed)
        fragments.append(fragment)
    return molecules_with_target_removed, fragments


def visualize_atom_removals(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: str,
    molecule_str: str,
    seed: int,
):
    """Generates visualizations of the predictions when removing each atom from a molecule."""
    molecule, molecule_name = analysis.construct_molecule(molecule_str)
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )

    # Remove the target atoms from the molecule.
    molecules_with_target_removed = []
    fragments = []
    for target in range(len(molecule)):
        molecule_with_target_removed = ase.Atoms(
            positions=np.concatenate(
                [molecule.positions[:target], molecule.positions[target + 1 :]]
            ),
            numbers=np.concatenate(
                [molecule.numbers[:target], molecule.numbers[target + 1 :]]
            ),
        )
        fragment = input_pipeline.ase_atoms_to_jraph_graph(
            molecule_with_target_removed,
            ATOMIC_NUMBERS,
            config.radial_cutoff,
        )

        molecules_with_target_removed.append(molecule_with_target_removed)
        fragments.append(fragment)

    # We don't actually need a PRNG key, since we're not sampling.
    logging.info("Computing predictions...")
    rng = jax.random.PRNGKey(seed)
    preds = jax.jit(model.apply)(
        params,
        rng,
        jraph.batch(fragments),
        focus_and_atom_type_inverse_temperature,
        position_inverse_temperature,
    )
    preds = jax.tree_util.tree_map(np.asarray, preds)
    preds = jraph.unbatch(preds)
    logging.info("Predictions computed.")

    # Create the output directory where HTML files will be saved.
    outputdir = os.path.join(
        outputdir,
        name,
        "visualizations",
        "atom_removal",
        f"inverse_temperature={focus_and_atom_type_inverse_temperature},{position_inverse_temperature}",
        f"step={step}",
    )
    os.makedirs(outputdir, exist_ok=True)

    # Loop over all possible targets.
    logging.info("Visualizing predictions...")
    figs = []
    for target in tqdm.tqdm(range(len(molecule)), desc="Targets"):
        # We have to remove the batch dimension.
        # Also, correct the focus indices due to batching.
        pred = preds[target]._replace(
            globals=jax.tree_util.tree_map(lambda x: np.squeeze(x, axis=0), preds[target].globals)
        )
        corrected_focus_indices = pred.globals.focus_indices - sum(
            p.n_node.item() for i, p in enumerate(preds) if i < target
        )
        pred = pred._replace(
            globals=pred.globals._replace(focus_indices=corrected_focus_indices)
        )

        # Visualize predictions for this target.
        fig = visualizer.visualize_predictions(
            pred, molecules_with_target_removed[target], molecule, target
        )

        figs.append(fig)

    # Combine all figures into one.
    fig_all = analysis.combine_visualizations(figs)

    # Add title.
    model_name = analysis.get_title_for_name(name)
    fig_all.update_layout(
        title=f"{model_name}: Predictions for {molecule_name}",
        title_x=0.5,
    )

    # Save to file.
    outputfile = os.path.join(
        outputdir,
        f"{molecule_name}_seed={seed}.html",
    )
    fig_all.write_html(outputfile, include_plotlyjs="cdn")

    return fig_all


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    outputdir = FLAGS.outputdir
    focus_and_atom_type_inverse_temperature = (
        FLAGS.focus_and_atom_type_inverse_temperature
    )
    position_inverse_temperature = FLAGS.position_inverse_temperature
    step = FLAGS.step
    molecule_str = FLAGS.molecule
    seed = FLAGS.seed

    visualize_atom_removals(
        workdir,
        outputdir,
        focus_and_atom_type_inverse_temperature,
        position_inverse_temperature,
        step,
        molecule_str,
        seed,
    )


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for model.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses"),
        "Directory where visualizations should be saved.",
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
        "molecule",
        None,
        "Molecule to use for experiment. Can be specified either as an index for the QM9 dataset, a name for ase.build.molecule(), or a file with atomic numbers and coordinates for ase.io.read().",
    )
    flags.DEFINE_integer(
        "seed",
        0,
        "PRNG seed for sampling.",
    )

    flags.mark_flags_as_required(["workdir", "molecule"])
    app.run(main)
