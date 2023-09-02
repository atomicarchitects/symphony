"""Visualize the fragments and corresponding predictions."""

from typing import Sequence
import os

from absl import flags
from absl import app
import jax
import jax.numpy as jnp
import numpy as np
import jraph

from symphony.data import input_pipeline_tf

from analyses import analysis
from analyses import visualizer

FLAGS = flags.FLAGS


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    outputdir = FLAGS.outputdir
    focus_and_atom_type_inverse_temperature = (
        FLAGS.focus_and_atom_type_inverse_temperature
    )
    position_inverse_temperature = FLAGS.position_inverse_temperature
    step = FLAGS.step

    visualize_predictions_and_fragments(
        workdir,
        outputdir,
        focus_and_atom_type_inverse_temperature,
        position_inverse_temperature,
        step,
    )


def visualize_predictions_and_fragments(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: str,
):
    """Visualize the predictions and fragments."""
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )

    # Load the dataset.
    # We disable shuffling to visualize step-by-step.
    config.shuffle_datasets = False
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, dataset_rng = jax.random.split(rng)
    datasets = input_pipeline_tf.get_datasets(dataset_rng, config)

    # Load the fragments and compute predictions.
    fragments = next(datasets["train"].take(1).as_numpy_iterator())
    preds = jax.jit(model.apply)(
        params,
        rng,
        fragments,
        focus_and_atom_type_inverse_temperature,
        position_inverse_temperature,
    )
    preds = jax.tree_map(np.asarray, preds)

    # Remove padding graphs.
    fragments = jraph.unpad_with_graphs(fragments)
    preds = jraph.unpad_with_graphs(preds)

    # We create one figure per fragment.
    figs = []
    for index, (fragment, pred) in enumerate(
        zip(jraph.unbatch(fragments), jraph.unbatch(preds))
    ):
        # Remove batch dimension.
        # Also, correct the focus indices.
        fragment = fragment._replace(
            globals=jax.tree_map(lambda x: np.squeeze(x, axis=0), fragment.globals)
        )
        pred = pred._replace(
            globals=jax.tree_map(lambda x: np.squeeze(x, axis=0), pred.globals)
        )
        corrected_focus_indices = (
            pred.globals.focus_indices - preds.n_node[:index].sum()
        )
        pred = pred._replace(
            globals=pred.globals._replace(focus_indices=corrected_focus_indices)
        )
        figs.append(visualizer.visualize_predictions(pred, fragment))

    # Save to files.
    visualizations_dir = os.path.join(
        outputdir,
        name,
        f"fait={focus_and_atom_type_inverse_temperature}",
        f"pit={position_inverse_temperature}",
        f"step={step}",
        "visualizations",
        "train_fragments",
    )
    os.makedirs(
        visualizations_dir,
        exist_ok=True,
    )
    for index in range(len(figs)):
        outputfile = os.path.join(
            visualizations_dir,
            f"fragments_{index}.html",
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
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
