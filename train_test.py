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
from configs import mace, e3schnet, nequip, marionette

try:
    import profile_nn_jax
except ImportError:
    profile_nn_jax = None

# Important to see the logging messages!
logging.getLogger().setLevel(logging.INFO)

_ALL_CONFIGS = {
    "e3schnet": e3schnet.get_config(),
    "mace": mace.get_config(),
    "nequip": nequip.get_config(),
    "marionette": marionette.get_config(),
}


def update_dummy_config(
    config: ml_collections.ConfigDict, train_on_split_smaller_than_chunk: bool
) -> ml_collections.FrozenConfigDict:
    """Updates the dummy config."""
    config = ml_collections.ConfigDict(config)
    config.num_train_steps = 100
    config.num_eval_steps = 10
    config.num_eval_steps_at_end_of_training = 10
    config.eval_every_steps = 50
    config.train_on_split_smaller_than_chunk = train_on_split_smaller_than_chunk
    if train_on_split_smaller_than_chunk:
        config.train_molecules = (0, 10)
    return ml_collections.FrozenConfigDict(config)


def create_dummy_data() -> Tuple[datatypes.Predictions, datatypes.Fragments]:
    """Creates dummy data for testing."""
    num_graphs = 2
    num_nodes = 5
    num_elements = models.NUM_ELEMENTS
    num_radii = models.RADII.shape[0]

    # Dummy predictions and graphs.
    position_coeffs = e3nn.IrrepsArray("0e", jnp.ones((num_graphs, num_radii, 1)))
    preds = datatypes.Predictions(
        nodes=datatypes.NodePredictions(
            focus_logits=jnp.ones((num_nodes,)), focus_probs=None, embeddings=None
        ),
        globals=datatypes.GlobalPredictions(
            stop=None,
            stop_probs=None,
            focus_indices=None,
            target_species_logits=jnp.asarray(
                [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.2, 1.3, 1.4, 1.5]]
            ),
            target_species_probs=None,
            target_species=None,
            position_coeffs=position_coeffs,
            position_logits=e3nn.to_s2grid(
                position_coeffs,
                res_beta=10,
                res_alpha=9,
                quadrature="gausslegendre",
                p_val=1,
                p_arg=-1,
            ),
            position_probs=None,
            position_vectors=None,
        ),
        edges=None,
        senders=None,
        receivers=None,
        n_node=jnp.asarray([2, 3]),
        n_edge=None,
    )
    graphs = datatypes.Fragments(
        nodes=datatypes.FragmentsNodes(
            positions=jnp.zeros((num_nodes, 3)),
            species=jnp.zeros((num_nodes,)),
            focus_probability=jnp.asarray([0.5, 0.5, 0.1, 0.1, 0.1]),
        ),
        globals=datatypes.FragmentsGlobals(
            stop=jnp.asarray([0, 0]),
            target_species=jnp.zeros((num_graphs,)),
            target_positions=jnp.ones((num_graphs, 3)),
            target_species_probability=jnp.ones((num_graphs, num_elements))
            / num_elements,
        ),
        edges=None,
        senders=None,
        receivers=None,
        n_node=jnp.asarray([2, 3]),
        n_edge=None,
    )
    return preds, graphs


class TrainTest(parameterized.TestCase):
    def setUp(self):
        self.preds, self.graphs = create_dummy_data()

    def test_focus_loss(self):
        _, (focus_loss, _, _) = train.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            radius_rbf_variance=30,
            target_position_inverse_temperature=1000,
        )
        expected_focus_loss = jnp.asarray(
            [-1 + jnp.log(1 + 2 * jnp.e), -0.3 + jnp.log(1 + 3 * jnp.e)]
        )
        self.assertSequenceAlmostEqual(focus_loss, expected_focus_loss, places=5)

    def test_atom_type_loss(self):
        _, (_, atom_type_loss, _) = train.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            radius_rbf_variance=30,
            target_position_inverse_temperature=1000,
        )
        expected_atom_type_loss = jnp.asarray(
            [
                -0.3 + scipy.special.logsumexp([0.1, 0.2, 0.3, 0.4, 0.5]),
                -1.3 + scipy.special.logsumexp([1.1, 1.2, 1.3, 1.4, 1.5]),
            ]
        )
        self.assertSequenceAlmostEqual(
            atom_type_loss, expected_atom_type_loss, places=5
        )

    @parameterized.parameters(1.0, 10.0, 100.0, 1000.0)
    def test_position_loss(self, target_position_inverse_temperature: float):
        _, (_, _, position_loss) = train.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            radius_rbf_variance=30,
            target_position_inverse_temperature=target_position_inverse_temperature,
        )
        target_positions = self.graphs.globals.target_positions
        position_logits = self.preds.globals.position_logits
        norms = jnp.linalg.norm(target_positions, axis=-1, keepdims=True)
        target_positions_unit_vectors = target_positions / jnp.where(
            norms == 0, 1, norms
        )
        target_positions_unit_vectors = e3nn.IrrepsArray(
            "1o", target_positions_unit_vectors
        )
        res_beta, res_alpha, quadrature = (
            position_logits.res_beta,
            position_logits.res_alpha,
            position_logits.quadrature,
        )
        log_true_angular_dist = e3nn.to_s2grid(
            target_position_inverse_temperature * target_positions_unit_vectors,
            res_beta,
            res_alpha,
            quadrature=quadrature,
            p_val=1,
            p_arg=-1,
        )
        log_true_angular_dist_max = jnp.max(
            log_true_angular_dist.grid_values, axis=(-2, -1), keepdims=True
        )
        true_angular_dist = log_true_angular_dist.apply(
            lambda x: jnp.exp(x - log_true_angular_dist_max)
        )
        true_angular_dist = true_angular_dist / true_angular_dist.integrate()

        log_true_angular_dist = true_angular_dist.apply(
            lambda x: jnp.log(jnp.where(x == 0, 1.0, x))
        )
        lower_bounds = (
            -(log_true_angular_dist * true_angular_dist).integrate().array.squeeze(-1)
        )
        num_radii = models.RADII.shape[0]

        expected_position_loss = (
            -1 + jnp.log(4 * jnp.pi * jnp.e * num_radii) - lower_bounds
        )

        self.assertTrue(jnp.all(position_loss >= 0))
        self.assertSequenceAlmostEqual(position_loss, expected_position_loss, places=4)

    @parameterized.product(
        config_name=["nequip"], train_on_split_smaller_than_chunk=[True]
    )
    def test_train_and_evaluate(
        self, config_name: str, train_on_split_smaller_than_chunk: bool
    ):
        """Tests that training and evaluation runs without errors."""
        # Ensure NaNs and Infs are detected.
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

        # Load config for dummy dataset.
        config = _ALL_CONFIGS[config_name]
        config = update_dummy_config(config, train_on_split_smaller_than_chunk)
        config = ml_collections.FrozenConfigDict(config)

        # Create a temporary directory where metrics are written.
        workdir = tempfile.mkdtemp()

        # Training should proceed without any errors.
        train.train_and_evaluate(config, workdir)

        # Save device memory profile.
        # jax.profiler.save_device_memory_profile(f"{config_name}.prof")


if __name__ == "__main__":
    absltest.main()
