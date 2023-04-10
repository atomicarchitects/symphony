"""Generates molecules from a trained model."""

from typing import Sequence

import os
import sys

from absl import flags
from absl import app
from absl import logging
import ase
import ase.data
import ase.io
import ase.visualize
import jax
import jax.numpy as jnp
import numpy as np
import jraph
import tqdm
import chex

sys.path.append("..")

import analyses.analysis as analysis
import analyses.visualize_atom_removals as visualize_atom_removals
import datatypes  # noqa: E402
import input_pipeline  # noqa: E402
import models  # noqa: E402


MAX_NUM_ATOMS = 10

FLAGS = flags.FLAGS


def generate_molecules(
    workdir: str, outputdir: str, beta: float, step: int, seeds: Sequence[int], visualize: bool
):
    """Generates molecules from a trained model at the given workdir."""

    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )
    apply_fn = jax.jit(model.apply)

    # Create output directory.
    step_name = "step=best" if step == -1 else f"step={step}"
    molecules_outputdir = os.path.join(outputdir, "molecules", "generated", name, f"beta={beta}", step_name)
    visualizations_outputdir = os.path.join(outputdir, "visualizations", "molecules", name, f"beta={beta}", step_name)
    os.makedirs(molecules_outputdir, exist_ok=True)
    os.makedirs(visualizations_outputdir, exist_ok=True)

    def get_predictions(
        fragment: jraph.GraphsTuple, rng: chex.PRNGKey
    ) -> datatypes.Predictions:
        fragments = jraph.pad_with_graphs(fragment, n_node=32, n_edge=1024, n_graph=2)
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
        molecule = ase.Atoms(
            positions=jnp.array([[0.0, 0.0, 0.0]]),
            numbers=jnp.array([6]),
        )

        rng = jax.random.PRNGKey(seed)
        for step in range(MAX_NUM_ATOMS):
            step_rng, rng = jax.random.split(rng)
            fragment = input_pipeline.ase_atoms_to_jraph_graph(
                molecule, models.ATOMIC_NUMBERS, config.nn_cutoff
            )

            # Run the model on the current molecule.
            pred = get_predictions(fragment, step_rng)

            # Check for any NaNs in the predictions.
            num_nans = sum(jax.tree_leaves(jax.tree_map(lambda x: jnp.isnan(x).sum(), pred)))
            if num_nans > 0:
                logging.info("NaNs in predictions. Stopping generation...")
                break

            # Check if we should stop.
            stop = pred.globals.stop.item()
            if stop:
                break

            # Save visualization of generation process.
            if visualize:
                fig = visualize_atom_removals.visualize_predictions(pred, molecule)
                model_name = visualize_atom_removals.get_title_for_name(name)
                fig.update_layout(
                    title=f"{model_name}: Predictions for Seed {seed} at Step {step}",
                    title_x=0.5,
                )

                # Save to file.
                outputfile = os.path.join(
                    visualizations_outputdir,
                    f"seed={seed}_step={step}.html",
                )
                fig.write_html(outputfile)

            # Append the new atom to the molecule.
            molecule = append_predictions(molecule, pred)

        # We don't generate molecules with more than MAX_NUM_ATOMS atoms.
        if molecule.numbers.shape[0] < 1000:
            logging.info("Generated %s", molecule.get_chemical_formula())
            ase.io.write(os.path.join(molecules_outputdir, f"seed={seed}_molecule.xyz"), molecule)
        else:
            logging.info("Discarding %s because it is too long", molecule.get_chemical_formula())



def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    outputdir = FLAGS.outputdir
    beta = FLAGS.beta
    step = FLAGS.step
    seeds = [int(seed) for seed in FLAGS.seeds]
    visualize = FLAGS.visualize
    
    generate_molecules(workdir, outputdir, beta, step, seeds, visualize)


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
    flags.DEFINE_bool(
        "visualize",
        False,
        "Whether to visualize the generation process step-by-step.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
