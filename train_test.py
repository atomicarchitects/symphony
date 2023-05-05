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
    config: ml_collections.ConfigDict, train_on_split_smaller_than_chunk: bool,
    position_loss_type: str
) -> ml_collections.FrozenConfigDict:
    """Updates the dummy config."""
    config = ml_collections.ConfigDict(config)
    config.num_train_steps = 100
    config.num_eval_steps = 10
    config.num_eval_steps_at_end_of_training = 10
    config.eval_every_steps = 50
    config.train_on_split_smaller_than_chunk = train_on_split_smaller_than_chunk
    config.loss_kwargs.position_loss_type = position_loss_type
    if train_on_split_smaller_than_chunk:
        config.train_molecules = (0, 10)
    return ml_collections.FrozenConfigDict(config)


def create_dummy_data() -> Tuple[datatypes.Predictions, datatypes.Fragments]:
    """Creates dummy data for testing."""
    num_graphs = 2
    num_nodes = 8
    num_elements = models.NUM_ELEMENTS
    n_node = jnp.asarray([num_nodes // 2, num_nodes // 2])

    # Dummy predictions and graphs.
    coeffs_array = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    coeffs_array = jnp.repeat(
        coeffs_array[:, None, :], repeats=len(models.RADII), axis=1
    )
    position_coeffs = e3nn.IrrepsArray("0e + 1o", coeffs_array)
    position_logits = e3nn.to_s2grid(
        position_coeffs,
        res_beta=180,
        res_alpha=359,
        quadrature="gausslegendre",
        normalization="integral",
        p_val=1,
        p_arg=-1,
    )
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
            position_logits=position_logits,
            position_probs=None,
            position_vectors=None,
        ),
        edges=None,
        senders=None,
        receivers=None,
        n_node=n_node,
        n_edge=None,
    )

    graphs = datatypes.Fragments(
        nodes=datatypes.FragmentsNodes(
            positions=jnp.zeros((num_nodes, 3)),
            species=jnp.zeros((num_nodes,)),
            focus_probability=jnp.asarray([0.5, 0.5, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0]),
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
        n_node=n_node,
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
            radius_rbf_variance=1e-3,
            target_position_inverse_temperature=1000,
            ignore_position_loss_for_small_fragments=True,
            position_loss_type="kl_divergence",
        )
        # sum(-qv * fv) + log(1 + sum(exp(fv)))
        expected_focus_loss = jnp.asarray(
            [-1 + jnp.log(1 + 4 * jnp.e), -0.3 + jnp.log(1 + 4 * jnp.e)]
        )
        self.assertSequenceAlmostEqual(focus_loss, expected_focus_loss, places=5)

    def test_atom_type_loss(self):
        _, (_, atom_type_loss, _) = train.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            radius_rbf_variance=1e-3,
            target_position_inverse_temperature=1000,
            ignore_position_loss_for_small_fragments=True,
            position_loss_type="kl_divergence",
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
            radius_rbf_variance=1e-3,
            target_position_inverse_temperature=target_position_inverse_temperature,
            ignore_position_loss_for_small_fragments=True,
            position_loss_type="kl_divergence",
        )

        # Precomputed self-entropies for the different inverse temperatures.
        SELF_ENTROPIES = {
            1.0: 4.02192,
            10.0: 1.8630707,
            100.0: -0.43951172,
            1000.0: -2.7420683,
        }
        self_entropy = SELF_ENTROPIES[target_position_inverse_temperature]

        # Since the predicted distribution is uniform, we can easily compute the expected position loss.
        num_radii = len(models.RADII)
        expected_position_loss = jnp.asarray(
            [
                -1 + jnp.log(4 * jnp.pi * jnp.e * num_radii) - self_entropy,
                -1 + jnp.log(4 * jnp.pi * jnp.e * num_radii) - self_entropy,
            ]
        )

        self.assertTrue(jnp.all(position_loss >= 0))
        self.assertSequenceAlmostEqual(position_loss, expected_position_loss, places=4)

    @parameterized.product(
        config_name=["nequip"],
        train_on_split_smaller_than_chunk=[False],
        position_loss_type=["l2", "kl_divergence"]
    )
    def test_train_and_evaluate(
        self, config_name: str, train_on_split_smaller_than_chunk: bool,
        position_loss_type: str
    ):
        """Tests that training and evaluation runs without errors."""
        # Ensure NaNs and Infs are detected.
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

        # Load config for dummy dataset.
        config = _ALL_CONFIGS[config_name]
        config = update_dummy_config(config, train_on_split_smaller_than_chunk, position_loss_type)
        config = ml_collections.FrozenConfigDict(config)

        # Create a temporary directory where metrics are written.
        workdir = tempfile.mkdtemp()

        # Training should proceed without any errors.
        train.train_and_evaluate(config, workdir)

        # Save device memory profile.
        # jax.profiler.save_device_memory_profile(f"{config_name}.prof")


if __name__ == "__main__":
    absltest.main()
