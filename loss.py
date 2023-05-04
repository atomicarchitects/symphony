import functools
from typing import Tuple

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import optax

import datatypes
import models


@functools.partial(jax.profiler.annotate_function, name="generation_loss")
def generation_loss(
    preds: datatypes.Predictions,
    graphs: datatypes.Fragments,
    radius_rbf_variance: float,
    target_position_inverse_temperature: float,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the loss for the generation task.
    Args:
        preds (datatypes.Predictions): the model predictions
        graphs (datatypes.Fragment): a batch of graphs representing the current molecules
    """
    num_radii = models.RADII.shape[0]
    num_graphs = graphs.n_node.shape[0]
    num_nodes = graphs.nodes.positions.shape[0]
    num_elements = models.NUM_ELEMENTS

    def focus_loss() -> jnp.ndarray:
        """Computes the loss over focus probabilities."""
        assert (
            preds.nodes.focus_logits.shape
            == graphs.nodes.focus_probability.shape
            == (num_nodes,)
        )

        n_node = graphs.n_node
        focus_logits = preds.nodes.focus_logits

        # Compute sum(qv * fv) for each graph,
        # where fv is the focus_logits for node v,
        # and qv is the focus_probability for node v.
        loss_focus = e3nn.scatter_sum(
            -graphs.nodes.focus_probability * focus_logits, nel=n_node
        )

        # This is basically log(1 + sum(exp(fv))) for each graph.
        # But we subtract out the maximum fv per graph for numerical stability.
        focus_logits_max = e3nn.scatter_max(focus_logits, nel=n_node, initial=0.0)
        focus_logits_max_expanded = e3nn.scatter_max(
            focus_logits, nel=n_node, map_back=True, initial=0.0
        )
        focus_logits -= focus_logits_max_expanded
        loss_focus += focus_logits_max + jnp.log(
            jnp.exp(-focus_logits_max)
            + e3nn.scatter_sum(jnp.exp(focus_logits), nel=n_node)
        )

        assert loss_focus.shape == (num_graphs,)
        return loss_focus

    def atom_type_loss() -> jnp.ndarray:
        """Computes the loss over atom types."""
        assert (
            preds.globals.target_species_logits.shape
            == graphs.globals.target_species_probability.shape
            == (num_graphs, num_elements)
        )

        loss_atom_type = optax.softmax_cross_entropy(
            logits=preds.globals.target_species_logits,
            labels=graphs.globals.target_species_probability,
        )

        assert loss_atom_type.shape == (num_graphs,)
        return loss_atom_type

    def position_loss() -> jnp.ndarray:
        """Computes the loss over position probabilities."""

        assert preds.globals.position_coeffs.array.shape == (
            num_graphs,
            num_radii,
            preds.globals.position_coeffs.irreps.dim,
        )

        position_logits = preds.globals.position_logits
        res_beta, res_alpha, quadrature = (
            position_logits.res_beta,
            position_logits.res_alpha,
            position_logits.quadrature,
        )

        def safe_log(x: jnp.ndarray) -> jnp.ndarray:
            """Computes the log of x, replacing 0 with 1 for numerical stability."""
            return jnp.log(jnp.where(x == 0, 1.0, x))

        def kl_divergence_on_spheres(
            true_radius_weights: jnp.ndarray,
            log_true_angular_coeffs: e3nn.IrrepsArray,
            log_predicted_dist: e3nn.SphericalSignal,
        ) -> jnp.ndarray:
            """Compute the KL divergence between two distributions on the sphere."""
            # Convert coefficients to a distribution on the sphere.
            log_true_angular_dist = e3nn.to_s2grid(
                log_true_angular_coeffs,
                res_beta,
                res_alpha,
                quadrature=quadrature,
                p_val=1,
                p_arg=-1,
            )

            # Subtract the maximum value for numerical stability.
            log_true_angular_dist_max = jnp.max(
                log_true_angular_dist.grid_values, axis=(-2, -1), keepdims=True
            )
            log_true_angular_dist = log_true_angular_dist.apply(
                lambda x: x - log_true_angular_dist_max
            )

            # Convert to a probability distribution, by taking the exponential and normalizing.
            true_angular_dist = log_true_angular_dist.apply(jnp.exp)
            true_angular_dist = true_angular_dist / true_angular_dist.integrate()

            # Check that shapes are correct.
            assert true_angular_dist.grid_values.shape == (
                res_beta,
                res_alpha,
            ), true_angular_dist.grid_values.shape
            assert true_radius_weights.shape == (num_radii,)

            # Mix in the radius weights to get a distribution over all spheres.
            true_dist = true_radius_weights * true_angular_dist[None, :, :]
            # Now, compute the unnormalized predicted distribution over all spheres.
            # Subtract the maximum value for numerical stability.
            log_predicted_dist_max = jnp.max(log_predicted_dist.grid_values)
            log_predicted_dist = log_predicted_dist.apply(
                lambda x: x - log_predicted_dist_max
            )

            # Compute the cross-entropy including a normalizing factor to account for the fact that the predicted distribution is not normalized.
            cross_entropy = -(true_dist * log_predicted_dist).integrate().array.sum()
            normalizing_factor = jnp.log(
                log_predicted_dist.apply(jnp.exp).integrate().array.sum()
            )

            # Compute the self-entropy of the true distribution.
            self_entropy = (
                -(true_dist * true_dist.apply(safe_log)).integrate().array.sum()
            )

            # This should be non-negative, upto numerical precision.
            return cross_entropy + normalizing_factor - self_entropy

        def target_position_to_radius_weights(
            target_position: jnp.ndarray,
        ) -> jnp.ndarray:
            """Returns the radial distribution for a target position."""
            radius_weights = jax.vmap(
                lambda radius: jnp.exp(
                    -((radius - jnp.linalg.norm(target_position)) ** 2)
                    / (2 * radius_rbf_variance)
                )
            )(models.RADII)
            radius_weights += 1e-10
            return radius_weights / jnp.sum(radius_weights)

        def target_position_to_log_angular_coeffs(
            target_position: jnp.ndarray,
        ) -> e3nn.IrrepsArray:
            """Returns the temperature-scaled angular distribution for a target position."""
            # Compute the true distribution over positions,
            # described by a smooth approximation of a delta function at the target positions.
            norm = jnp.linalg.norm(target_position, axis=-1, keepdims=True)
            target_position_unit_vector = target_position / jnp.where(
                norm == 0, 1, norm
            )
            target_position_unit_vector = e3nn.IrrepsArray(
                "1o", target_position_unit_vector
            )
            return target_position_inverse_temperature * target_position_unit_vector

        target_positions = graphs.globals.target_positions
        true_radius_weights = jax.vmap(target_position_to_radius_weights)(
            target_positions
        )
        log_true_angular_coeffs = jax.vmap(target_position_to_log_angular_coeffs)(
            target_positions
        )
        log_predicted_dist = position_logits

        loss_position = jax.vmap(kl_divergence_on_spheres)(
            true_radius_weights, log_true_angular_coeffs, log_predicted_dist
        )
        assert loss_position.shape == (num_graphs,)
        return loss_position

    # If this is the last step in the generation process, we do not have to predict atom type and position.
    loss_focus = focus_loss()
    loss_atom_type = atom_type_loss() * (1 - graphs.globals.stop)
    loss_position = position_loss() * (1 - graphs.globals.stop)

    # Ignore position loss for graphs with less than, or equal to 3 atoms.
    loss_position = jnp.where(graphs.n_node <= 3, 0, loss_position)

    total_loss = loss_focus + loss_atom_type + loss_position
    return total_loss, (
        loss_focus,
        loss_atom_type,
        loss_position,
    )
