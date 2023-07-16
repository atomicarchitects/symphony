"""Main file for running the training pipeline.

This file is intentionally kept short.
The majority for logic is in libraries that can be easily tested.
"""

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
import ml_collections
from ml_collections import config_flags
import tensorflow as tf


from symphony import train
from symphony.models import models
from configs import root_dirs


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    # We only support single-host training on a single device.
    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
    )

    # Freeze config.
    config = FLAGS.config
    config.root_dir = root_dirs.get_root_dir(config.dataset, config.fragment_logic)
    config.loss_kwargs.min_radius = config.target_position_predictor.min_radius
    config.loss_kwargs.max_radius = config.target_position_predictor.max_radius
    config.loss_kwargs.num_radii = config.target_position_predictor.num_radii
    config = ml_collections.FrozenConfigDict(config)

    # Start training!
    train.train_and_evaluate(config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
