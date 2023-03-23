"""Tests for the training loop."""

from typing import Tuple
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import scipy
import ml_collections
import logging

import models
import datatypes
import train
from configs import mace, e3schnet, nequip

try:
    import profile_nn_jax
except ImportError:
    profile_nn_jax = None

logging.getLogger().setLevel(logging.INFO)  # Important to see the messages!

_ALL_CONFIGS = {
    "e3schnet": e3schnet.get_config(),
    "mace": mace.get_config(),
    "nequip": nequip.get_config(),
}


def update_dummy_config(config):
    """Updates the dummy config."""
    config.num_train_steps = 100
    config.num_eval_steps = 10
    config.num_eval_steps_at_end_of_training = 10
    config.eval_every_steps = 50


def create_dummy_data() -> Tuple[datatypes.Predictions, datatypes.Fragments]:
    """Creates dummy data for testing."""
    num_graphs = 2
    num_nodes = 5
    num_elements = models.NUM_ELEMENTS
    num_radii = models.RADII.shape[0]
    position_coeffs = e3nn.IrrepsArray("0e", jnp.ones((num_graphs, num_radii, 1)))
    preds = datatypes.Predictions(
        focus_logits=jnp.ones((num_nodes,)),
        target_species_logits=jnp.asarray(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.2, 1.3, 1.4, 1.5]]
        ),
        position_coeffs=position_coeffs,
        position_logits=e3nn.to_s2grid(
            position_coeffs,
            res_beta=10,
            res_alpha=9,
            quadrature="gausslegendre",
            p_val=1,
            p_arg=-1,
        ),
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
            target_positions=jnp.zeros((num_graphs, 3)),
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

    def test_position_loss(self):
        _, (_, _, position_loss) = train.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            radius_rbf_variance=30,
        )
        num_radii = models.RADII.shape[0]
        expected_position_loss = jnp.asarray(
            [
                -1 + jnp.log(4 * jnp.pi * jnp.e * num_radii),
                -1 + jnp.log(4 * jnp.pi * jnp.e * num_radii),
            ]
        )
        self.assertSequenceAlmostEqual(position_loss, expected_position_loss, places=4)

    @parameterized.parameters("mace", "e3schnet", "nequip")
    def test_train_and_evaluate(self, config_name: str):
        """Tests that training and evaluation runs without errors."""

        # Ensure NaNs and Infs are detected.
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

        # Load config for dummy dataset.
        config = _ALL_CONFIGS[config_name]
        update_dummy_config(config)
        config = ml_collections.FrozenConfigDict(config)

        # Create a temporary directory where metrics are written.
        workdir = tempfile.mkdtemp()

        # Training should proceed without any errors.
        train.train_and_evaluate(config, workdir)


if __name__ == "__main__":
    absltest.main()
