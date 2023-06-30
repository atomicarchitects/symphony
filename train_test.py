"""Tests for the training loop."""

from typing import Tuple
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import e3nn_jax as e3nn
import jax
import jax.profiler
import jax.numpy as jnp
import scipy
import ml_collections
import logging

import models
import datatypes
import train
import loss_test
from configs.qm9 import mace, e3schnet, nequip, marionette
from configs.tetris import nequip as tetris_nequip
from configs.platonic_solids import nequip as platonic_solids_nequip
from configs import root_dirs

# Important to see the logging messages!
logging.getLogger().setLevel(logging.INFO)

_ALL_CONFIGS = {
    "qm9": {
        "e3schnet": e3schnet.get_config(),
        "mace": mace.get_config(),
        "nequip": nequip.get_config(),
        "marionette": marionette.get_config(),
    },
    "tetris": {"nequip": tetris_nequip.get_config()},
    "platonic_solids": {"nequip": platonic_solids_nequip.get_config()},
}


def update_dummy_config(
    config: ml_collections.ConfigDict,
    train_on_split_smaller_than_chunk: bool,
    position_loss_type: str,
    dataset: str,
) -> ml_collections.FrozenConfigDict:
    """Updates the dummy config."""
    config.num_train_steps = 100
    config.num_eval_steps = 10
    config.num_eval_steps_at_end_of_training = 10
    config.eval_every_steps = 50
    config.loss_kwargs.position_loss_type = position_loss_type
    config.dataset = dataset
    config.root_dir = root_dirs.get_root_dir(config.dataset, config.fragment_logic)
    if dataset == "qm9":
        config.train_on_split_smaller_than_chunk = train_on_split_smaller_than_chunk
        if train_on_split_smaller_than_chunk:
            config.train_molecules = (0, 10)
    return ml_collections.FrozenConfigDict(config)


class TrainTest(parameterized.TestCase):
    def setUp(self):
        self.preds, self.graphs = loss_test.create_dummy_data()

    @parameterized.product(
        config_name=["nequip"],
        train_on_split_smaller_than_chunk=[True],
        position_loss_type=["kl_divergence"],
        dataset=["tetris"],
    )
    def test_train_and_evaluate(
        self,
        config_name: str,
        train_on_split_smaller_than_chunk: bool,
        position_loss_type: str,
        dataset: str,
    ):
        """Tests that training and evaluation runs without errors."""
        # self.skipTest("This test is too slow.")

        # Ensure NaNs and Infs are detected.
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

        # Load config for dummy dataset.
        config = _ALL_CONFIGS[dataset][config_name]
        config = update_dummy_config(
            config, train_on_split_smaller_than_chunk, position_loss_type, dataset
        )
        config = ml_collections.FrozenConfigDict(config)

        # Create a temporary directory where metrics are written.
        workdir = tempfile.mkdtemp()

        # Training should proceed without any errors.
        train.train_and_evaluate(config, workdir)

        # Save device memory profile.
        # jax.profiler.save_device_memory_profile(f"profiles/{config_name}.prof")

    @parameterized.product(
        config_name=["nequip", "mace", "e3schnet", "marionette"],
        rng=[0, 1],
    )
    def test_equivariance(self, config_name: str, rng: int):
        """Tests that models are equivariant."""
        self.skipTest("This test is too slow.")

        rng = jax.random.PRNGKey(rng)
        config = _ALL_CONFIGS["qm9"][config_name]
        model = models.create_model(config, run_in_evaluation_mode=False)
        params = model.init(rng, self.graphs)

        def apply_fn(positions: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            """Wraps the model's apply function."""
            graphs = self.graphs._replace(
                nodes=self.graphs.nodes._replace(positions=positions.array)
            )
            return model.apply(params, rng, graphs).globals.position_coeffs

        input_positions = e3nn.IrrepsArray("1o", self.graphs.nodes.positions)
        e3nn.utils.assert_equivariant(apply_fn, rng_key=rng, args_in=(input_positions,))


if __name__ == "__main__":
    absltest.main()
