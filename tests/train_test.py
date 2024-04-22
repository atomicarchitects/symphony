"""Tests for the training loop."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import e3nn_jax as e3nn
import jax
import jax.profiler
import ml_collections
import logging

from symphony import models, train
from . import loss_test

from configs import root_dirs
from configs.qm9 import test as qm9_test
from configs.platonic_solids import test as platonic_solids_test
from configs.geom_drugs import test as geom_drugs_test

# Important to see the logging messages!
logging.getLogger().setLevel(logging.INFO)

_ALL_CONFIGS = {
    "qm9_test": qm9_test.get_config(),
    "platonic_solids_test": platonic_solids_test.get_config(),
    "geom_drugs_test": geom_drugs_test.get_config(),
}


class TrainTest(parameterized.TestCase):
    @parameterized.product(
        config_name=["qm9_test"]
    )
    def test_train_and_evaluate(
        self,
        config_name: str,
    ):
        """Tests that training and evaluation runs without errors."""
        # Ensure NaNs and Infs are detected.
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

        # Load config for dummy dataset.
        config = _ALL_CONFIGS[config_name]
        config.root_dir = root_dirs.get_root_dir(config.dataset)
        config = ml_collections.FrozenConfigDict(config)
    
        # Create a temporary directory where metrics are written.
        workdir = tempfile.mkdtemp()

        # Training should proceed without any errors.
        train.train_and_evaluate(config, workdir)

        # Save device memory profile.
        # jax.profiler.save_device_memory_profile(f"profiles/{config_name}.prof")


    @parameterized.product(
        config_name=["qm9_test"],
        rng=[0, 1, 2],
    )
    def test_equivariance(self, config_name: str, rng: int):
        """Tests that models are equivariant."""
        self.skipTest("This test is too slow.")
        config = _ALL_CONFIGS[config_name]
        config = ml_collections.FrozenConfigDict(config)

        graphs, _ = loss_test.create_dummy_data()

        rng = jax.random.PRNGKey(rng)
        model = models.create_model(config, run_in_evaluation_mode=False)
        params = model.init(rng, graphs)

        def apply_fn(positions: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
            """Wraps the model's apply function."""
            graphs = graphs._replace(
                nodes=graphs.nodes._replace(positions=positions.array)
            )
            return model.apply(params, rng, graphs).globals.log_position_coeffs

        input_positions = e3nn.IrrepsArray("1o", graphs.nodes.positions)
        e3nn.utils.assert_equivariant(apply_fn, rng, input_positions, atol=2e-4)


if __name__ == "__main__":
    absltest.main()
