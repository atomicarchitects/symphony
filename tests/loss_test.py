"""Tests for the loss.""" ""

from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized

from absl.testing import absltest
from absl.testing import parameterized
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np

from symphony import datatypes, loss, models


def create_dummy_data() -> Tuple[datatypes.Predictions, datatypes.Fragments]:
    """Creates dummy data for testing."""
    num_graphs = 2
    num_elements = models.NUM_ELEMENTS
    n_node = jnp.asarray([2, 3])
    num_nodes = jnp.sum(n_node)

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
            focus_and_target_species_logits=jnp.asarray(
                [
                    [jnp.log(2), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
                    [jnp.log(2), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
                    [jnp.log(2), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
                    [jnp.log(1), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
                    [jnp.log(3), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
                ]
            ),
            focus_and_target_species_probs=None,
            embeddings=None,
            auxiliary_node_embeddings=None,
        ),
        globals=datatypes.GlobalPredictions(
            stop_logits=jnp.asarray([0.0, 0.0]),
            stop_probs=None,
            stop=None,
            focus_indices=None,
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
            positions=jax.random.normal(jax.random.PRNGKey(0), (num_nodes, 3)),
            species=jnp.zeros((num_nodes,), dtype=jnp.int32),
            focus_and_target_species_probs=jnp.asarray(
                [
                    [0.1, 0.0, 0.5, 0.0, 0.0],
                    [0.1, 0.1, 0.0, 0.2, 0.0],
                    [0.2, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.5],
                    [0.0, 0.0, 0.0, 0.3, 0.0],
                ]
            ),
        ),
        globals=datatypes.FragmentsGlobals(
            stop=jnp.asarray([0, 0]),
            target_species=jnp.zeros((num_graphs,), dtype=jnp.int32),
            target_positions=jnp.ones((num_graphs, 3)),
        ),
        edges=None,
        senders=jnp.asarray([0, 1, 2, 3, 2]),
        receivers=jnp.asarray([1, 0, 2, 3, 4]),
        n_node=n_node,
        n_edge=None,
    )
    return preds, graphs


class LossTest(parameterized.TestCase):
    def setUp(self):
        self.preds, self.graphs = create_dummy_data()

    def test_focus_and_atom_type_loss(self):
        _, (
            focus_and_atom_type_loss,
            _,
        ) = loss.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            radius_rbf_variance=1e-3,
            target_position_inverse_temperature=1000,
            target_position_lmax=1,
            ignore_position_loss_for_small_fragments=False,
            position_loss_type="kl_divergence",
        )
        logits = self.preds.nodes.focus_and_target_species_logits
        stop_logits = self.preds.globals.stop_logits
        targets = self.graphs.nodes.focus_and_target_species_probs
        stop_targets = self.graphs.globals.stop

        n_node = self.graphs.n_node
        num_nodes = jnp.sum(n_node)
        segment_ids = models.get_segment_ids(n_node, num_nodes)
        num_segments = len(n_node)
        probs, stop_probs = models.segment_softmax_2D_with_stop(
            logits, self.preds.globals.stop_logits, segment_ids, num_segments
        )

        expected_focus_and_atom_type_loss = -jnp.asarray(
            [
                (targets * loss.safe_log(probs))[:2, :].sum()
                + (stop_targets * loss.safe_log(stop_probs))[0],
                (targets * loss.safe_log(probs))[2:, :].sum()
                + (stop_targets * loss.safe_log(stop_probs))[1],
            ]
        )
        expected_focus_and_atom_type_loss_logits = -jnp.asarray(
            [
                (targets * logits)[:2, :].sum()
                + (stop_targets * stop_logits)[0]
                - jnp.log(21),
                (targets * logits)[2:, :].sum()
                + (stop_targets * stop_logits)[1]
                - jnp.log(31),
            ]
        )
        lower_bounds = -jnp.asarray(
            [
                (targets * loss.safe_log(targets))[:2, :].sum()
                + (stop_targets * loss.safe_log(stop_targets))[0],
                (targets * loss.safe_log(targets))[2:, :].sum()
                + (stop_targets * loss.safe_log(stop_targets))[1],
            ]
        )

        np.testing.assert_allclose(
            expected_focus_and_atom_type_loss_logits,
            expected_focus_and_atom_type_loss,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            focus_and_atom_type_loss,
            expected_focus_and_atom_type_loss - lower_bounds,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            focus_and_atom_type_loss,
            expected_focus_and_atom_type_loss_logits - lower_bounds,
            atol=1e-5,
        )

    def compute_self_entropy_of_true_position_distribution(
        self,
        target_position: jnp.ndarray,
        radius_rbf_variance: float,
        target_position_inverse_temperature: float,
        lmax: int,
    ) -> jnp.ndarray:
        """Compute the self-entropy of the true position distribution, which is uniform over all space."""
        position_logits = self.preds.globals.position_logits
        res_beta, res_alpha, quadrature = (
            position_logits.res_beta,
            position_logits.res_alpha,
            position_logits.quadrature,
        )

        # Compute coefficients for the true angular distribution.
        log_true_angular_coeffs = loss.target_position_to_log_angular_coeffs(
            target_position, target_position_inverse_temperature, lmax
        )

        # Convert coefficients to a distribution on the sphere.
        log_true_angular_dist = e3nn.to_s2grid(
            log_true_angular_coeffs,
            res_beta=res_beta,
            res_alpha=res_alpha,
            quadrature=quadrature,
            p_val=1,
            p_arg=-1,
        )

        # Subtract the maximum value for numerical stability.
        log_true_angular_dist_max = jnp.max(
            log_true_angular_dist.grid_values, axis=(-2, -1), keepdims=True
        )
        log_true_angular_dist_max = jax.lax.stop_gradient(log_true_angular_dist_max)
        log_true_angular_dist = log_true_angular_dist.apply(
            lambda x: x - log_true_angular_dist_max
        )

        # Convert to a probability distribution, by taking the exponential and normalizing.
        true_angular_dist = log_true_angular_dist.apply(jnp.exp)
        true_angular_dist = true_angular_dist / true_angular_dist.integrate()

        # Mix in the radius weights to get a distribution over all spheres.
        true_radius_weights = loss.target_position_to_radius_weights(
            target_position, radius_rbf_variance
        )
        true_dist = true_radius_weights * true_angular_dist[None, :, :]

        # Compute the self-entropy of the true distribution.
        self_entropy = (
            -(true_dist * true_dist.apply(loss.safe_log)).integrate().array.sum()
        )
        return self_entropy

    @parameterized.parameters(1.0, 10.0, 100.0, 1000.0)
    def test_kl_divergence_position_loss(
        self, target_position_inverse_temperature: float
    ):
        _, (_, position_loss) = loss.generation_loss(
            preds=self.preds,
            graphs=self.graphs,
            radius_rbf_variance=1e-3,
            target_position_inverse_temperature=target_position_inverse_temperature,
            target_position_lmax=1,
            ignore_position_loss_for_small_fragments=False,
            position_loss_type="kl_divergence",
        )

        # Compute self-entropy.
        self_entropies = jax.vmap(
            lambda pos: self.compute_self_entropy_of_true_position_distribution(
                pos,
                radius_rbf_variance=1e-3,
                target_position_inverse_temperature=target_position_inverse_temperature,
                lmax=self.preds.globals.position_coeffs.irreps.lmax,
            )
        )(self.graphs.globals.target_positions)

        # Since the predicted distribution is uniform, we can easily compute the expected position loss.
        num_radii = len(models.RADII)
        expected_position_loss = (
            -1 + jnp.log(4 * jnp.pi * jnp.e * num_radii) - self_entropies
        )

        self.assertTrue(jnp.all(position_loss >= 0))
        np.testing.assert_allclose(position_loss, expected_position_loss, atol=1e-4)

    def test_logits_shift(self):
        preds = self.preds._replace(
            globals=self.preds.globals._replace(
                position_logits=self.preds.globals.position_logits.apply(lambda x: x + 1),
            )
        )

        _, (_, position_loss) = loss.generation_loss(
            preds=preds,
            graphs=self.graphs,
            radius_rbf_variance=1e-3,
            target_position_inverse_temperature=1.0,
            target_position_lmax=1,
            ignore_position_loss_for_small_fragments=False,
            position_loss_type="kl_divergence",
        )

        preds_2 = self.preds._replace(
            globals=self.preds.globals._replace(
                position_logits=self.preds.globals.position_logits.apply(lambda x: x + 2),
            )
        )

        _, (_, position_loss_2) = loss.generation_loss(
            preds=preds_2,
            graphs=self.graphs,
            radius_rbf_variance=1e-3,
            target_position_inverse_temperature=1.0,
            target_position_lmax=1,
            ignore_position_loss_for_small_fragments=False,
            position_loss_type="kl_divergence",
        )

        np.testing.assert_allclose(position_loss, position_loss_2, atol=1e-4)



if __name__ == "__main__":
    absltest.main()
