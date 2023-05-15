"""Creates a series of visualizations to build up a molecule."""
import os
import pickle
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

import analyses.analysis as analysis  # noqa: E402
import input_pipeline  # noqa: E402
import models  # noqa: E402

FLAGS = flags.FLAGS

ATOMIC_NUMBERS = models.ATOMIC_NUMBERS
ELEMENTS = ["H", "C", "N", "O", "F"]
RADII = models.RADII

# Colors and sizes for the atoms.
ATOMIC_COLORS = {
    1: "rgb(150, 150, 150)",  # H
    6: "rgb(50, 50, 50)",  # C
    7: "rgb(0, 100, 255)",  # N
    8: "rgb(255, 0, 0)",  # O
    9: "rgb(255, 0, 255)",  # F
}
ATOMIC_SIZES = {
    1: 10,  # H
    6: 30,  # C
    7: 30,  # N
    8: 30,  # O
    9: 30,  # F
}

def _remove_target_atoms(molecule: ase.Atoms, cutoff: float) -> Tuple[List[ase.Atoms], List[jraph.GraphsTuple]]:
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


def visualize_atom_removals_against_steps(
    workdir: str,
    outputdir: str,
    beta: float,
    steps: Sequence[int],
    target: int,
    molecule_str: str,
    use_cache: bool,
    seed: int,
):
    """Generates visualizations of the predictions when removing each atom from a molecule."""
    # Remove the target atom
    molecule, molecule_name = analysis.construct_molecule(molecule_str)
    name = analysis.name_from_workdir(workdir)

    molecules_with_target_removed = None
    figs = []
    for step in steps:
        model, params, config = analysis.load_model_at_step(
            workdir, step, run_in_evaluation_mode=True
        )
        if molecules_with_target_removed is None:
            molecules_with_target_removed, fragments = _remove_target_atoms(molecule, cutoff=config.nn_cutoff)

        # We don't actually need a PRNG key, since we're not sampling.
        logging.info("Computing predictions...")
        rng = jax.random.PRNGKey(seed)
        preds = jax.jit(model.apply)(params, rng, jraph.batch(fragments), beta)
        preds = jax.tree_map(np.asarray, preds)
        preds = jraph.unbatch(preds)
        logging.info("Predictions computed.")

        # Create the output directory where HTML files will be saved.
        step_name = "step=best" if step == -1 else f"step={step}"
        outputdir = os.path.join(
            outputdir,
            "visualizations",
            "atom_removal",
            name,
            f"beta={beta}",
            step_name,
        )
        os.makedirs(outputdir, exist_ok=True)

        # Loop over all possible targets.
        logging.info("Visualizing predictions...")
        # We have to remove the batch dimension.
        # Also, correct the focus indices due to batching.
        pred = preds[target]._replace(
            globals=jax.tree_map(lambda x: np.squeeze(x, axis=0), preds[target].globals)
        )
        corrected_focus_indices = pred.globals.focus_indices - sum(
            p.n_node.item() for i, p in enumerate(preds) if i < target
        )
        pred = pred._replace(
            globals=pred.globals._replace(focus_indices=corrected_focus_indices)
        )

        # Visualize predictions for this target.
        fig = analysis.visualize_predictions(
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
        f"{molecule_name}_seed={seed}_against_steps.html",
    )
    fig_all.write_html(outputfile, include_plotlyjs="cdn")

    return fig_all


def visualize_atom_removals(
    workdir: str,
    outputdir: str,
    beta: float,
    step: int,
    molecule_str: str,
    use_cache: bool,
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
            config.nn_cutoff,
        )

        molecules_with_target_removed.append(molecule_with_target_removed)
        fragments.append(fragment)

    # We don't actually need a PRNG key, since we're not sampling.
    logging.info("Computing predictions...")
    preds_path = os.path.join(
        f"cached/{workdir.replace('/', '_')}_{molecule_name}_preds.pkl"
    )
    if use_cache and os.path.exists(preds_path):
        logging.info("Using cached predictions at %s", os.path.abspath(preds_path))
        preds = pickle.load(open(preds_path, "rb"))
    else:
        rng = jax.random.PRNGKey(seed)
        preds = jax.jit(model.apply)(params, rng, jraph.batch(fragments), beta)
        preds = jax.tree_map(np.asarray, preds)
        preds = jraph.unbatch(preds)
        os.makedirs(os.path.dirname(preds_path), exist_ok=True)
        pickle.dump(preds, open(preds_path, "wb"))
        logging.info("Predictions computed.")

    # Create the output directory where HTML files will be saved.
    step_name = "step=best" if step == -1 else f"step={step}"
    outputdir = os.path.join(
        outputdir,
        "visualizations",
        "atom_removal",
        name,
        f"beta={beta}",
        step_name,
    )
    os.makedirs(outputdir, exist_ok=True)

    # Loop over all possible targets.
    logging.info("Visualizing predictions...")
    figs = []
    for target in tqdm.tqdm(range(len(molecule)), desc="Targets"):
        # We have to remove the batch dimension.
        # Also, correct the focus indices due to batching.
        pred = preds[target]._replace(
            globals=jax.tree_map(lambda x: np.squeeze(x, axis=0), preds[target].globals)
        )
        corrected_focus_indices = pred.globals.focus_indices - sum(
            p.n_node.item() for i, p in enumerate(preds) if i < target
        )
        pred = pred._replace(
            globals=pred.globals._replace(focus_indices=corrected_focus_indices)
        )

        # Visualize predictions for this target.
        fig = analysis.visualize_predictions(
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
    beta = FLAGS.beta
    step = FLAGS.step
    molecule_str = FLAGS.molecule
    use_cache = FLAGS.use_cache
    seed = FLAGS.seed

    visualize_atom_removals(
        workdir, outputdir, beta, step, molecule_str, use_cache, seed
    )


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for model.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses"),
        "Directory where visualizations should be saved.",
    )
    flags.DEFINE_float("beta", 1.0, "Inverse temperature value for sampling.")
    flags.DEFINE_integer(
        "step",
        -1,
        "Step number to load model from. The default of -1 corresponds to the best model.",
    )
    flags.DEFINE_string(
        "molecule",
        None,
        "Molecule to use for experiment. Can be specified either as an index for the QM9 dataset, a name for ase.build.molecule(), or a file with atomic numbers and coordinates for ase.io.read().",
    )
    flags.DEFINE_bool(
        "use_cache",
        False,
        "Whether to use cached predictions if they exist.",
    )
    flags.DEFINE_integer(
        "seed",
        0,
        "PRNG seed for sampling.",
    )

    flags.mark_flags_as_required(["workdir", "molecule"])
    app.run(main)
