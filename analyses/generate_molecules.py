"""Generates molecules from a trained model."""

from typing import List, Sequence

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
import numpy as np
import jraph
import tqdm
import chex
import pickle
from plotly import graph_objects as go
from plotly.subplots import make_subplots

sys.path.append("..")

import analyses.analysis as analysis
import analyses.visualize_atom_removals as visualize_atom_removals
import datatypes  # noqa: E402
import input_pipeline  # noqa: E402
import models  # noqa: E402


MAX_NUM_ATOMS = 30

FLAGS = flags.FLAGS


def nan_analysis(pred: datatypes.Predictions):
    global_attrs = [
        "stop_probs",
        "stop",
        "focus_indices",
        "target_species_logits",
        "target_species_probs",
        "target_species",
        "position_coeffs",
        "position_logits",
        "position_probs",
        "position_vectors",
    ]
    for attr in global_attrs:
        attr_value = getattr(pred.globals, attr)
        try:
            num_nans = jnp.isnan(attr_value).sum()
        except:
            try:
                num_nans = jnp.isnan(attr_value.array).sum()
            except:
                num_nans = jnp.isnan(attr_value.grid_values).sum()
        print(f"{attr} has {num_nans} nans")


def generate_molecules(
    workdir: str,
    outputdir: str,
    beta: float,
    step: int,
    seeds: Sequence[int],
    init_molecule: str,
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
    apply_fn = jax.jit(model.apply)
    logging.info(config.to_dict())

    # Create output directories.
    step_name = "step=best" if step == -1 else f"step={step}"
    molecules_outputdir = os.path.join(
        outputdir, "molecules", "generated", name, f"beta={beta}", step_name
    )
    os.makedirs(molecules_outputdir, exist_ok=True)
    if visualize:
        visualizations_outputdir = os.path.join(
            outputdir, "visualizations", "molecules", name, f"beta={beta}", step_name
        )
        os.makedirs(visualizations_outputdir, exist_ok=True)
    molecule_list = []

    def get_predictions(
        fragment: jraph.GraphsTuple, rng: chex.PRNGKey
    ) -> datatypes.Predictions:
        fragments = jraph.pad_with_graphs(fragment, n_node=80, n_edge=4096, n_graph=2)
        preds = apply_fn(params, rng, fragments, beta)

        # Remove the batch dimension.
        pred = jraph.unpad_with_graphs(preds)
        pred = pred._replace(
            globals=jax.tree_map(lambda x: np.squeeze(x, axis=0), pred.globals)
        )
        return pred

    def append_predictions(
        molecule: ase.Atoms, pred: datatypes.Predictions
    ) -> ase.Atoms:
        focus = pred.globals.focus_indices
        pos_focus = molecule.positions[focus]
        pos_rel = pred.globals.position_vectors

        new_species = jnp.array(
            models.ATOMIC_NUMBERS[pred.globals.target_species.item()]
        )
        new_position = pos_focus + pos_rel

        return ase.Atoms(
            positions=jnp.concatenate(
                [molecule.positions, new_position[None, :]], axis=0
            ),
            numbers=jnp.concatenate([molecule.numbers, new_species[None]], axis=0),
        )

    # Generate with different seeds.
    for seed in tqdm.tqdm(seeds, desc="Generating molecules"):
        rng = jax.random.PRNGKey(seed)
        molecule = init_molecule.copy()
        nan_found = False

        # Add atoms step-by-step.
        figs = []
        for step in range(MAX_NUM_ATOMS):
            step_rng, rng = jax.random.split(rng)
            fragment = input_pipeline.ase_atoms_to_jraph_graph(
                molecule, models.ATOMIC_NUMBERS, config.nn_cutoff
            )

            # Run the model on the current molecule.
            pred = get_predictions(fragment, step_rng)

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
                break

            # Save visualization of generation process.
            if visualize:
                fig = analysis.visualize_predictions(pred, molecule)
                figs.append(fig)

            # Check if we should stop.
            stop = pred.globals.stop.item()
            if stop:
                break

            # Append the new atom to the molecule.
            molecule = append_predictions(molecule, pred)

        # We don't generate molecules with more than MAX_NUM_ATOMS atoms.
        if molecule.numbers.shape[0] < 1000:
            logging.info("Generated %s", molecule.get_chemical_formula())
            if nan_found:
                outputfile = f"{init_molecule_name}_seed={seed}_NaN.xyz"
            else:
                outputfile = f"{init_molecule_name}_seed={seed}.xyz"
            ase.io.write(os.path.join(molecules_outputdir, outputfile), molecule)
            molecule_list.append(molecule)
        else:
            logging.info(
                "Discarding %s because it is too long", molecule.get_chemical_formula()
            )

        if visualize:
            # Combine visualizations.
            fig_all = analysis.combine_visualizations(figs)

            # Add title.
            model_name = analysis.get_title_for_name(name)
            fig_all.update_layout(
                title=f"{model_name}: Predictions for Seed {seed}",
                title_x=0.5,
            )

            # Save to file.
            outputfile = os.path.join(
                visualizations_outputdir,
                f"{init_molecule_name}_seed={seed}.html",
            )
            fig_all.write_html(outputfile, include_plotlyjs="cdn")

    # Save the generated molecules as an ASE database.
    output_db = os.path.join(molecules_outputdir, f"generated_molecules_init={init_molecule_name}.db")
    with connect(output_db) as conn:
        for mol in molecule_list:
            conn.write(mol)
    ase_to_mol_dict(molecule_list, file_name=os.path.join(molecules_outputdir, "generated_molecules.mol_dict"))


def ase_to_mol_dict(molecules: List[ase.Atoms], save=True, file_name=None):
    """from G-SchNet: https://github.com/atomistic-machine-learning/G-SchNet"""

    generated = {}
    for mol in molecules:
        l = mol.get_atomic_numbers().shape[0]
        if l not in generated:
            generated[l] = {
                "_positions": np.array([mol.get_positions()]),
                "_atomic_numbers": np.array([mol.get_atomic_numbers()]),
            }
        else:
            generated[l]["_positions"] = np.append(
                generated[l]["_positions"],
                np.array([mol.get_positions()]),
                0,
            )
            generated[l]["_atomic_numbers"] = np.append(
                generated[l]["_atomic_numbers"],
                np.array([mol.get_atomic_numbers()]),
                0,
            )

    if save:
        with open(file_name, "wb") as f:
            pickle.dump(generated, f)

    return generated



def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    outputdir = FLAGS.outputdir
    beta = FLAGS.beta
    step = FLAGS.step
    seeds = [int(seed) for seed in FLAGS.seeds]
    init = FLAGS.init
    visualize = FLAGS.visualize

    generate_molecules(workdir, outputdir, beta, step, seeds, init, visualize)


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for model.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses"),
        "Directory where molecules should be saved.",
    )
    flags.DEFINE_float("beta", 1.0, "Inverse temperature value for sampling.")
    flags.DEFINE_integer(
        "step",
        -1,
        "Step number to load model from. The default of -1 corresponds to the best model.",
    )
    flags.DEFINE_list(
        "seeds",
        list(range(64)),
        "Seeds to attempt to generate molecules from.",
    )
    flags.DEFINE_string(
        "init",
        "C",
        "An initial molecular fragment to start the generation process from.",
    )
    flags.DEFINE_bool(
        "visualize",
        False,
        "Whether to visualize the generation process step-by-step.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)

