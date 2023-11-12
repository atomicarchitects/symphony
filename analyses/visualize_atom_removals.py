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

FLAGS = flags.FLAGS


def visualize_atom_removals(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: str,
    init_molecule: str,
    seed: int,
    num_atoms_to_remove: int,
):
    """Generates visualizations of the predictions when removing each atom from a molecule."""
    molecule, molecule_name = analysis.construct_molecule(init_molecule)
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )

    # Remove the target atoms from the molecule.
    fragments = []
    for target in range(len(molecule) - num_atoms_to_remove + 1):
        molecule_with_target_removed = ase.Atoms(
            positions=np.concatenate(
                [molecule.positions[:target], molecule.positions[target + num_atoms_to_remove :]]
            ),
            numbers=np.concatenate(
                [molecule.numbers[:target], molecule.numbers[target + num_atoms_to_remove :]]
            ),
        )
        fragment = input_pipeline.ase_atoms_to_jraph_graph(
            molecule_with_target_removed,
            atomic_numbers=np.asarray([1, 6, 7, 8, 9]),
            nn_cutoff=config.nn_cutoff,
        )
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
    preds = jax.tree_map(np.asarray, preds)
    preds = jraph.unbatch(preds)
    logging.info("Predictions computed.")

    # Create the output directory where HTML files will be saved.
    outputdir = os.path.join(
        outputdir,
        name,
        f"fait={focus_and_atom_type_inverse_temperature}",
        f"pit={position_inverse_temperature}",
        f"step={step}",
        "visualizations",
        "atom_removal",
    )
    os.makedirs(outputdir, exist_ok=True)

    # Loop over all possible targets.
    logging.info("Visualizing predictions...")
    figs = []
    for target in tqdm.tqdm(range(len(molecule) - num_atoms_to_remove + 1), desc="Targets"):
        fig = visualizer.visualize_predictions(
            preds[target], fragments[target],
        )
        if num_atoms_to_remove == 1:
            title = f"Predictions for {molecule_name}: Target {target} removed"
        else:
            title = f"Predictions for {molecule_name}: Targets {target} to {target + num_atoms_to_remove - 1} removed"
        fig.update_layout(
            title=title,
            title_x=0.5,
        )

        outputfile = os.path.join(
            outputdir,
            f"{molecule_name}_num_removed={num_atoms_to_remove}_seed={seed}_target={target}.html",
        )
        fig.write_html(outputfile, include_plotlyjs="cdn")
        figs.append(fig)
    return figs

def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)

    visualize_atom_removals(
        workdir,
        FLAGS.outputdir,
        FLAGS.focus_and_atom_type_inverse_temperature,
        FLAGS.position_inverse_temperature,
        FLAGS.step,
        FLAGS.init,
        FLAGS.seed,
        FLAGS.num_atoms_to_remove,
    )


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for model.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses", "analysed_workdirs"),
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
        "init",
        None,
        "Molecule to use for experiment. Can be specified either as an index for the QM9 dataset, a name for ase.build.molecule(), or a file with atomic numbers and coordinates for ase.io.read().",
    )
    flags.DEFINE_integer(
        "seed",
        0,
        "PRNG seed for sampling.",
    )
    flags.DEFINE_integer(
        "num_atoms_to_remove",
        1,
        "Number of atoms to remove from the molecule.",
    )

    flags.mark_flags_as_required(["workdir", "init"])
    app.run(main)
