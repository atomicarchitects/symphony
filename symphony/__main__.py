"""Main file for running the training pipeline.

This file is intentionally kept short.
The majority for logic is in libraries that can be easily tested.
"""
import os

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
import ml_collections
from ml_collections import config_flags
import tensorflow as tf


from symphony import train, train_position_updater
from configs import root_dirs


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_string("dataset", "qm9", "Dataset to train on.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Make sure the dataloader is deterministic.
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    # We only support single-host training on a single device.
    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())
    logging.info("CUDA_VISIBLE_DEVICES: %r", os.environ.get("CUDA_VISIBLE_DEVICES"))

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
    config = ml_collections.FrozenConfigDict(config)

    # Start training!
    if config.get("position_updater"):
        train_position_updater.train_and_evaluate(config, FLAGS.workdir)
    else:
        train.train_and_evaluate(config, FLAGS.workdir, dataset=FLAGS.dataset)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
