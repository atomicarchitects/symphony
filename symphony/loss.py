from typing import Tuple

import jax
import jax.numpy as jnp
import jraph
import functools
import e3nn_jax as e3nn

from symphony import datatypes, models


@jax.profiler.annotate_function
def generation_loss(
    preds: datatypes.Predictions,
    graphs: datatypes.Fragments,
    ignore_position_loss_for_small_fragments: bool,
    discretized_loss: bool,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Computes the loss for the generation task.
    Args:
    - preds: the model predictions
    - graphs: a batch of graphs representing the current molecules
    - ignore_position_loss_for_small_fragments: whether to ignore the position loss for fragments with less than or equal to 3 atoms
    """
    num_targets = graphs.globals.target_positions_mask.shape[-1]
    num_graphs = graphs.n_node.shape[0]
    num_nodes = graphs.nodes.positions.shape[0]
    n_node = graphs.n_node
    segment_ids = models.get_segment_ids(n_node, num_nodes)

    def focus_and_atom_type_loss() -> jnp.ndarray:
        """Computes the loss over focus and atom types for all nodes."""
        species_logits = preds.nodes.focus_and_target_species_logits
        species_targets = graphs.nodes.focus_and_target_species_probs
        stop_logits = preds.globals.stop_logits
        stop_targets = graphs.globals.stop.astype(jnp.float32)

        assert species_logits.shape == (num_nodes, species_logits.shape[-1])
        assert species_targets.shape == (num_nodes, species_logits.shape[-1])
        assert stop_logits.shape == (num_graphs,)
        assert stop_targets.shape == (num_graphs,)

        # Subtract the maximum value for numerical stability.
        # This doesn't affect the forward pass, nor the backward pass.
        logits_max = jraph.segment_max(
            species_logits, segment_ids, num_segments=num_graphs
        ).max(axis=-1)
        logits_max = jnp.maximum(logits_max, stop_logits)
        logits_max = jax.lax.stop_gradient(logits_max)
        species_logits -= logits_max[segment_ids, None]
        stop_logits -= logits_max

        # Compute the cross-entropy loss.
        loss_focus_and_atom_type = -(species_targets * species_logits).sum(axis=-1)
        loss_focus_and_atom_type = jraph.segment_sum(
            loss_focus_and_atom_type, segment_ids, num_graphs
        )
        loss_focus_and_atom_type += -stop_targets * stop_logits
        loss_focus_and_atom_type += jnp.log(
            jraph.segment_sum(
                jnp.exp(species_logits).sum(axis=-1), segment_ids, num_graphs
            )
            + jnp.exp(stop_logits)
        )

        # Compute the lower bound on cross-entropy loss as the entropy of the target distribution.
        lower_bounds = -(species_targets * models.safe_log(species_targets)).sum(
            axis=-1
        )
        lower_bounds = jraph.segment_sum(lower_bounds, segment_ids, num_graphs)
        lower_bounds += -stop_targets * models.safe_log(stop_targets)
        lower_bounds = jax.lax.stop_gradient(lower_bounds)

        # Subtract out self-entropy (lower bound) to get the KL divergence.
        loss_focus_and_atom_type -= lower_bounds
        assert loss_focus_and_atom_type.shape == (num_graphs,)

        return loss_focus_and_atom_type

    def process_logits(logits):
        target_positions_mask = graphs.globals.target_positions_mask
        assert target_positions_mask.shape == (num_graphs, num_targets)

        loss = -logits
        loss = jnp.where(target_positions_mask, loss, 0)
        loss = loss.sum(axis=-1)
        num_valid_targets = jnp.maximum(1, target_positions_mask.sum(axis=-1))
        loss /= num_valid_targets

        assert loss.shape == (num_graphs,)
        return loss

    def position_loss() -> jnp.ndarray:
        """Computes the loss over position probabilities."""
        assert graphs.globals.target_positions.shape == (num_graphs, num_targets, 3), (
            graphs.globals.target_positions.shape,
            num_graphs,
            num_targets,
            3,
        )
        position_logits = preds.globals.radial_logits + preds.globals.angular_logits
        assert position_logits.shape == (num_graphs, num_targets), (
            position_logits.shape,
            num_graphs,
            num_targets,
        )

        return (process_logits(preds.globals.radial_logits),
            process_logits(preds.globals.angular_logits),
            process_logits(position_logits))
    
    def discretized_position_loss() -> jnp.ndarray:
        """Computes the loss over position probabilities using separate losses for the radial and the angular components."""

        def target_position_to_radius_weights(
            target_position: jnp.ndarray, radii: jnp.ndarray, radius_rbf_variance: float = 1e-5
        ) -> jnp.ndarray:
            """Returns the radial distribution for a target position."""
            radial_logits = -(
                (radii - jnp.tile(jnp.linalg.norm(target_position, axis=-1), radii.shape[0]))
                ** 2
            )
            radial_logits /= 2 * radius_rbf_variance
            radial_weights = jax.nn.softmax(radial_logits, axis=-1)
            return radial_weights
        
        def target_position_to_log_angular_coeffs(
            target_position: jnp.ndarray,
            lmax: int,
            target_position_inverse_temperature: float = 20.0,
        ) -> e3nn.IrrepsArray:
            """Returns the temperature-scaled angular distribution for a target position."""
            # Compute the true distribution over positions,
            # described by a smooth approximation of a delta function at the target positions.
            norm = jnp.linalg.norm(target_position, axis=-1, keepdims=True)
            target_position_unit_vector = target_position / jnp.where(norm == 0, 1, norm)
            return target_position_inverse_temperature * e3nn.s2_dirac(
                target_position_unit_vector,
                lmax=lmax,
                p_val=1,
                p_arg=-1,
            )
        
        def kl_divergence_on_spheres(
            true_dist: e3nn.SphericalSignal,
            log_predicted_dist: e3nn.SphericalSignal,
        ) -> jnp.ndarray:
            """Compute the KL divergence between two distributions on the spheres."""
            assert true_dist.shape == log_predicted_dist.shape, (
                true_dist.shape,
                log_predicted_dist.shape,
            )

            # Now, compute the unnormalized predicted distribution over all spheres.
            # Subtract the maximum value for numerical stability.
            log_predicted_dist_max = jnp.max(log_predicted_dist.grid_values)
            log_predicted_dist_max = jax.lax.stop_gradient(log_predicted_dist_max)
            log_predicted_dist = log_predicted_dist.apply(lambda x: x - log_predicted_dist_max)

            # Compute the cross-entropy including a normalizing factor to account for the fact that the predicted distribution is not normalized.
            cross_entropy = -(true_dist * log_predicted_dist).integrate().array.sum()
            normalizing_factor = jnp.log(
                log_predicted_dist.apply(jnp.exp).integrate().array.sum()
            )

            # Compute the self-entropy of the true distribution.
            self_entropy = (
                -(true_dist * true_dist.apply(models.safe_log)).integrate().array.sum()
            )

            # This should be non-negative, upto numerical precision.
            return cross_entropy + normalizing_factor - self_entropy

        def kl_divergence_for_radii(
            true_radial_weights: jnp.ndarray, predicted_radial_logits: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute the KL divergence between two distributions on the radii."""
            return (
                true_radial_weights
                * (models.safe_log(true_radial_weights) - predicted_radial_logits)
            ).sum()

        # print("Angular logits shape:", preds.globals.angular_logits.shape)
        # print("Radial logits shape:", preds.globals.radial_logits.shape)
        quadrature = "gausslegendre"
        _, num_radii = preds.globals.radial_logits.shape
        radial_bins = jnp.linspace(0, 5, num_radii)  # TODO hardcoded
        lmax = 5  # TODO hardcoded
        res_beta = 100  # TODO hardcoded
        res_alpha = 99  # TODO hardcoded

        target_positions = graphs.globals.target_positions
        target_positions_mask = graphs.globals.target_positions_mask
        true_radial_weights = jax.vmap(jax.vmap(
            lambda pos: target_position_to_radius_weights(
                pos, radial_bins
            )
        ))(target_positions)
        log_true_angular_coeffs = jax.vmap(jax.vmap(
            lambda pos: target_position_to_log_angular_coeffs(
                pos, lmax
            )
        ))(target_positions)
        compute_joint_distribution_fn = functools.partial(
            models.compute_grid_of_joint_distribution,
            res_beta=res_beta,
            res_alpha=res_alpha,
            quadrature=quadrature,
        )
        true_angular_dist = jax.vmap(jax.vmap(
            compute_joint_distribution_fn,
        ))(
            jnp.ones((num_graphs, num_targets, 1),),
            log_true_angular_coeffs,
        )  # (num_graphs, max_targets_per_graph, 1, res_beta, res_alpha)
        true_angular_dist.grid_values = true_angular_dist.grid_values[:, :, 0, :, :]
        target_positions_mask = target_positions_mask[:, :, None, None]
        num_target_positions = jnp.sum(
            target_positions_mask, axis=1,
        )
        num_target_positions = jnp.maximum(num_target_positions, 1.0)
        angular_dist_sum = jnp.sum(
            true_angular_dist.grid_values * target_positions_mask, axis=1
        )
        angular_dist_mean = angular_dist_sum[:, :, :] / num_target_positions
        assert angular_dist_mean.shape == (num_graphs, res_beta, res_alpha)
        mean_true_angular_dist = e3nn.SphericalSignal(
            grid_values=angular_dist_mean, quadrature=true_angular_dist.quadrature
        )
        assert mean_true_angular_dist.shape == (num_graphs, res_beta, res_alpha)

        sum_radial_weights = jnp.sum(
            true_radial_weights * target_positions_mask[:, :, :, 0],
            axis=1,
        )
        mean_radial_weights = sum_radial_weights / num_target_positions[:, :, 0]
        radial_normalization = mean_radial_weights.sum(axis=-1, keepdims=True)
        mean_radial_weights /= jnp.where(
            radial_normalization == 0, 1.0, radial_normalization
        )
        assert mean_radial_weights.shape == (
            num_graphs,
            num_radii,
        )

        predicted_radial_logits = preds.globals.radial_logits
        # if predicted_radial_logits is None:
        #     position_logits = preds.globals.position_logits
        #     position_probs = jax.vmap(models.position_logits_to_position_distribution)(
        #         position_logits
        #     )

        #     predicted_radial_dist = jax.vmap(
        #         models.position_distribution_to_radial_distribution,
        #     )(position_probs)
        #     predicted_radial_logits = models.safe_log(predicted_radial_dist)

        assert (
            predicted_radial_logits.shape == mean_radial_weights.shape
        ), (predicted_radial_logits.shape, mean_radial_weights.shape)
        # jax.debug.print("Max predicted logit: {x}", x=jnp.max(predicted_radial_logits))
        # jax.debug.print("Min predicted logit: {x}", x=jnp.min(predicted_radial_logits))
        # jax.debug.print("True radial weights: {x}", x=mean_radial_weights)
        # jax.debug.print("Predicted radial logits: {x}", x=predicted_radial_logits)

        loss_radial = jax.vmap(kl_divergence_for_radii)(
            mean_radial_weights,
            jax.nn.log_softmax(predicted_radial_logits),
        )

        predicted_angular_logits = preds.globals.angular_logits
        assert predicted_angular_logits.shape == (num_graphs, num_targets), (
            predicted_angular_logits.shape,
            num_graphs,
            num_targets,
        )

        target_positions_mask = graphs.globals.target_positions_mask
        assert target_positions_mask.shape == (num_graphs, num_targets)

        loss_angular = -predicted_angular_logits
        loss_angular = jnp.where(target_positions_mask, loss_angular, 0)
        loss_angular = loss_angular.sum(axis=-1)
        num_valid_targets = jnp.maximum(1, target_positions_mask.sum(axis=-1))
        loss_angular /= num_valid_targets

        assert loss_angular.shape == (num_graphs,)

        loss_position = loss_angular + loss_radial
        assert loss_position.shape == (num_graphs,)
        # jax.debug.print("Radial loss: {x}", x=loss_radial)
        # jax.debug.print("Angular loss: {x}", x=loss_angular)

        return loss_radial, loss_angular, loss_position

    # If we should predict a STOP for this fragment, we do not have to predict a position.
    loss_focus_and_atom_type = focus_and_atom_type_loss()
    if discretized_loss:
        loss_radial, loss_angular, loss_position = discretized_position_loss()
    else:
        loss_radial, loss_angular, loss_position = position_loss()
    loss_radial = (1 - graphs.globals.stop) * loss_radial
    loss_angular = (1 - graphs.globals.stop) * loss_angular
    loss_position = (1 - graphs.globals.stop) * loss_position

    # Ignore position loss for graphs with less than, or equal to 3 atoms?
    # This is because there are symmetry-based degeneracies in the target distribution for these graphs.
    if ignore_position_loss_for_small_fragments:
        loss_position = jnp.where(n_node <= 3, 0, loss_position)

    total_loss = loss_focus_and_atom_type + loss_position
    return total_loss, (loss_focus_and_atom_type, loss_radial, loss_angular, loss_position)
