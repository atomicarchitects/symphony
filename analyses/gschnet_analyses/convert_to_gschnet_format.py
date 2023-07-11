"""Script to convert model predictions to the format used by analysis scripts in GSchNet."""

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp

import sys

sys.path.append("..")

import analyses.analysis as analysis
import datatypes
import input_pipeline_tf

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory where model data was stored.")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Load model.
    (
        config,
        best_state,
        best_state_in_eval_mode,
        metrics_for_best_state,
    ) = analysis.load_from_workdir(FLAGS.workdir)

    key = jax.random.PRNGKey(config.rng_seed)

    qm9_datasets = input_pipeline_tf.get_datasets(key, config)

    generated_dict = analysis.to_mol_dict(
        qm9_datasets["test"], "workdirs", "generated_single_frag"
    )


if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
