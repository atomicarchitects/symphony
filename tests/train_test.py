"""Tests for the training loop."""

from typing import Tuple
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import e3nn_jax as e3nn
import jax
import jax.profiler
import ml_collections
import logging

from symphony import models, train, train_position_updater

from configs.qm9 import mace, e3schnet, nequip, marionette, position_updater
from configs.tetris import nequip as tetris_nequip
from configs.platonic_solids import nequip as platonic_solids_nequip
from configs.platonic_solids import e3schnet_and_nequip as platonic_solids_e3schnet_and_nequip
from configs.platonic_solids import e3schnet_and_mace as platonic_solids_e3schnet_and_mace

from configs import root_dirs

# Important to see the logging messages!
logging.getLogger().setLevel(logging.INFO)

_ALL_CONFIGS = {
    "qm9": {
        "e3schnet": e3schnet.get_config(),
        "mace": mace.get_config(),
        "nequip": nequip.get_config(),
        "marionette": marionette.get_config(),
        "position_updater": position_updater.get_config(),
    },
    "tetris": {"nequip": tetris_nequip.get_config()},
    "platonic_solids": {
        "nequip": platonic_solids_nequip.get_config(),
        "e3schnet_and_nequip": platonic_solids_e3schnet_and_nequip.get_config(),
        "e3schnet_and_mace": platonic_solids_e3schnet_and_mace.get_config()
    },
}


def update_dummy_config(
    config: ml_collections.ConfigDict,
    train_on_split_smaller_than_chunk: bool,
    dataset: str,
) -> ml_collections.FrozenConfigDict:
    """Updates the dummy config."""
    config.num_train_steps = 1000
    config.num_eval_steps = 10
    config.log_every_steps = 100
    config.num_eval_steps_at_end_of_training = 10
    config.eval_every_steps = 500
    config.dataset = dataset
    config.root_dir = root_dirs.get_root_dir(config.dataset, config.fragment_logic)
    if dataset == "qm9":
        config.train_on_split_smaller_than_chunk = train_on_split_smaller_than_chunk
        if train_on_split_smaller_than_chunk:
            config.train_molecules = (0, 10)
    return ml_collections.FrozenConfigDict(config)


class TrainTest(parameterized.TestCase):
    """Tests for the training loop."""
    @parameterized.product(
        config_name=["e3schnet_and_mace"],
        train_on_split_smaller_than_chunk=[True],
        dataset=["platonic_solids"],
    )
    def test_train_and_evaluate(
        self,
        config_name: str,
        train_on_split_smaller_than_chunk: bool,
        dataset: str,
    ):
        """Tests that training and evaluation runs without errors."""
        # self.skipTest("This test is too slow.")
        # Ensure NaNs and Infs are detected.
        # jax.config.update("jax_debug_nans", True)
        # jax.config.update("jax_debug_infs", True)

        # Load config for dummy dataset.
        config = _ALL_CONFIGS[dataset][config_name]
        config = update_dummy_config(
            config, train_on_split_smaller_than_chunk, dataset
        )
        config = ml_collections.FrozenConfigDict(config)

        # Create a temporary directory where metrics are written.
        workdir = tempfile.mkdtemp()

        # Training should proceed without any errors.
        train.train_and_evaluate(config, workdir)

        # Save device memory profile.
        # jax.profiler.save_device_memory_profile(f"profiles/{config_name}.prof")


    @parameterized.product(
        config_name=["e3schnet_and_mace"],
        rng=[0, 1],
    )
    def test_equivariance(self, config_name: str, rng: int):
        """Tests that models are equivariant."""
        self.skipTest("This test is too slow.")
        rng = jax.random.PRNGKey(rng)
        config = _ALL_CONFIGS["platonic_solids"][config_name]
        model = models.create_model(config, run_in_evaluation_mode=False)
        params = model.init(rng, self.graphs)

        def apply_fn(positions: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            """Wraps the model's apply function."""
            graphs = self.graphs._replace(
                nodes=self.graphs.nodes._replace(positions=positions.array)
            )
            return model.apply(params, rng, graphs).globals.position_coeffs

        input_positions = e3nn.IrrepsArray("1o", self.graphs.nodes.positions)
        e3nn.utils.assert_equivariant(apply_fn, rng, input_positions, atol=1e-5)


if __name__ == "__main__":
    absltest.main()
