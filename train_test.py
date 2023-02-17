"""Tests for the training loop."""

"""Tests for train."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np

import train
from configs import graphmlp, graphnet, haikugraphmlp

_ALL_CONFIGS = {
    "graphmlp": graphmlp.get_config(),
    "graphnet": graphnet.get_config(),
    "haikugraphmlp": haikugraphmlp.get_config(),
}


def update_dummy_config(config):
    """Updates the dummy config."""
    config.dataset = "dummy"
    config.batch_size = 10
    config.max_degree = 2
    config.num_train_steps = 10
    config.num_classes = 10


class TrainTest(parameterized.TestCase):
    @parameterized.parameters("graphnet", "graphmlp", "haikugraphmlp")
    def test_train_and_evaluate(self, config_name: str):
        # Load config for dummy dataset.
        config = _ALL_CONFIGS[config_name]
        update_dummy_config(config)

        # Create a temporary directory where metrics are written.
        workdir = tempfile.mkdtemp()

        # Training should proceed without any errors.
        train.train_and_evaluate(config, workdir)


if __name__ == "__main__":
    absltest.main()
