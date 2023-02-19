"""Tests for the training loop."""

from typing import Tuple
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import e3nn_jax as e3nn
import jax.numpy as jnp
import scipy
import ml_collections

import models
import datatypes
import train
from configs import graphmlp, graphnet, haikugraphmlp, haikumace

_ALL_CONFIGS = {
    "graphmlp": graphmlp.get_config(),
    "graphnet": graphnet.get_config(),
    "haikugraphmlp": haikugraphmlp.get_config(),
    "haikumace": haikumace.get_config(),
}


def update_dummy_config(config):
    """Updates the dummy config."""
    config.batch_size = 10
    config.num_train_steps = 5


def _create_dummy_data() -> Tuple[datatypes.Predictions, datatypes.Fragment]:
    """Creates dummy data for testing."""
    num_graphs = 2
    num_nodes = 5
    num_elements = models.NUM_ELEMENTS
    num_radii = models.RADII.shape[0]
    preds = datatypes.Predictions(
        focus_logits=jnp.ones((num_nodes,)),
        species_logits=jnp.asarray(
            [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.2, 1.3, 1.4, 1.5]]
        ),
        position_coeffs=e3nn.IrrepsArray("0e", jnp.ones((num_graphs, num_radii, 1))),
    )
    graphs = datatypes.Fragment(
        nodes=datatypes.FragmentNodes(
            positions=jnp.zeros((num_nodes, 3)),
            species=jnp.zeros((num_nodes,)),
            focus_probability=jnp.asarray([0.5, 0.5, 0.1, 0.1, 0.1]),
        ),
        globals=datatypes.FragmentGlobals(
            stop=jnp.asarray([0, 1]),
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
        self.preds, self.graphs = _create_dummy_data()

    def test_focus_loss(self):
        _, (focus_loss, _, _) = train.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            res_beta=10,
            res_alpha=9,
            radius_rbf_variance=30,
        )
        expected_focus_loss = jnp.asarray(
            [-1 + jnp.log(1 + 2 * jnp.e), -0.3 + jnp.log(1 + 3 * jnp.e)]
        )
        print(focus_loss, expected_focus_loss)
        self.assertSequenceAlmostEqual(focus_loss, expected_focus_loss, places=5)

    def test_atom_type_loss(self):
        _, (_, atom_type_loss, _) = train.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            res_beta=10,
            res_alpha=9,
            radius_rbf_variance=30,
        )
        expected_atom_type_loss = jnp.asarray(
            [
                -0.3 + scipy.special.logsumexp([0.1, 0.2, 0.3, 0.4, 0.5]),
                -1.3 + scipy.special.logsumexp([1.1, 1.2, 1.3, 1.4, 1.5]),
            ]
        )
        self.assertSequenceAlmostEqual(atom_type_loss, expected_atom_type_loss)

    def test_position_loss(self):
        _, (_, _, position_loss) = train.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            res_beta=10,
            res_alpha=9,
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

    @parameterized.parameters("graphnet", "graphmlp", "haikugraphmlp", "haikumace")
    def test_train_and_evaluate(self, config_name: str):
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
