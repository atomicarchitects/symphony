import functools
from typing import Tuple, Optional

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np
import jraph
import ott

from symphony import datatypes, models


def safe_norm(x: jnp.ndarray, axis: int, eps: float = 1e-9) -> jnp.ndarray:
    """Computes the norm of x, replacing 0 with a small value (1e-9) for numerical stability."""
    x2 = jnp.sum(x**2, axis=axis)
    return jnp.sqrt(jnp.where(x2 == 0, eps, x2))


def target_position_to_radius_weights(
    target_position: jnp.ndarray, radius_rbf_variance: float, radii: jnp.ndarray
) -> jnp.ndarray:
    """Returns the radial distribution for a target position."""
    radial_logits = -((radii - jnp.linalg.norm(target_position)) ** 2)
    radial_logits /= 2 * radius_rbf_variance
    radial_weights = jax.nn.softmax(radial_logits, axis=-1)
    return radial_weights


def target_position_to_log_angular_coeffs(
    target_position: jnp.ndarray,
    target_position_inverse_temperature: float,
    lmax: int,
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


def l2_loss_on_spheres(
    log_true_dist_coeffs: e3nn.SphericalSignal,
    log_predicted_dist_coeffs: e3nn.SphericalSignal,
):
    """Computes the L2 loss between the logits of two distributions on the spheres."""
    assert log_true_dist_coeffs.shape == log_predicted_dist_coeffs.shape, (
        log_true_dist_coeffs.shape,
        log_predicted_dist_coeffs.shape,
    )

    norms_of_differences = e3nn.norm(
        log_true_dist_coeffs - log_predicted_dist_coeffs,
        per_irrep=False,
        squared=True,
    ).array.squeeze(-1)

    return jnp.sum(norms_of_differences)


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


def earthmover_distance_for_radii(
    true_radial_weights: jnp.ndarray,
    predicted_radial_logits: jnp.ndarray,
    radial_bins: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the Earthmover's distance between the logits of two distributions on the radii."""
    predicted_radial_weights = jax.nn.softmax(predicted_radial_logits, axis=-1)

    geom = ott.geometry.grid.Grid(x=[radial_bins])
    predicted_radial_weights = jnp.where(
        predicted_radial_weights == 0, 1e-9, predicted_radial_weights
    )
    prob = ott.problems.linear.linear_problem.LinearProblem(
        geom, a=predicted_radial_weights, b=true_radial_weights
    )
    solver = ott.solvers.linear.sinkhorn.Sinkhorn(lse_mode=True)
    out = solver(prob)
    return out.reg_ot_cost


@jax.profiler.annotate_function
def generation_loss(
    preds: datatypes.Predictions,
    graphs: datatypes.Fragments,
    radius_rbf_variance: float,
    target_position_inverse_temperature: float,
    target_position_lmax: Optional[int],
    ignore_position_loss_for_small_fragments: bool,
    position_loss_type: str,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Computes the loss for the generation task.
    Args:
        preds (datatypes.Predictions): the model predictions
        graphs (datatypes.Fragment): a batch of graphs representing the current molecules
    """
    radial_bins = preds.globals.radial_bins[0]  # Assume all radii are the same.
    num_radii = radial_bins.shape[0]
    num_graphs = graphs.n_node.shape[0]
    num_nodes = graphs.nodes.positions.shape[0]
    n_node = graphs.n_node
    segment_ids = models.get_segment_ids(n_node, num_nodes)
    if target_position_lmax is None:
        lmax = preds.globals.log_position_coeffs.irreps.lmax
    else:
        lmax = target_position_lmax

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

    def position_loss_with_kl_divergence() -> jnp.ndarray:
        """Computes the loss over position probabilities using the KL divergence."""

        position_logits = preds.globals.position_logits
        res_beta, res_alpha, quadrature = (
            position_logits.res_beta,
            position_logits.res_alpha,
            position_logits.quadrature,
        )

        target_positions = graphs.globals.target_positions
        true_radial_weights = jax.vmap(
            lambda pos: target_position_to_radius_weights(
                pos, radius_rbf_variance, radial_bins
            )
        )(target_positions)
        log_true_angular_coeffs = jax.vmap(
            lambda pos: target_position_to_log_angular_coeffs(
                pos, target_position_inverse_temperature, lmax
            )
        )(target_positions)
        compute_joint_distribution_fn = functools.partial(
            models.compute_grid_of_joint_distribution,
            res_beta=res_beta,
            res_alpha=res_alpha,
            quadrature=quadrature,
        )
        true_dist = jax.vmap(compute_joint_distribution_fn)(
            true_radial_weights, log_true_angular_coeffs
        )
        log_predicted_dist = position_logits

        assert true_radial_weights.shape == (
            num_graphs,
            num_radii,
        ), true_radial_weights.shape
        assert log_true_angular_coeffs.shape == (
            num_graphs,
            log_true_angular_coeffs.irreps.dim,
        )
        assert log_predicted_dist.grid_values.shape == (
            num_graphs,
            num_radii,
            res_beta,
            res_alpha,
        )

        loss_position = jax.vmap(kl_divergence_on_spheres)(
            true_dist, log_predicted_dist
        )
        assert loss_position.shape == (num_graphs,)

        return loss_position

    def position_loss_with_l2() -> jnp.ndarray:
        """Computes the loss over position probabilities using the L2 loss on the logits."""

        log_position_coeffs = preds.globals.log_position_coeffs
        target_positions = graphs.globals.target_positions
        true_radial_weights = jax.vmap(
            lambda pos: target_position_to_radius_weights(
                pos, radius_rbf_variance, radial_bins
            )
        )(target_positions)
        log_true_angular_coeffs = jax.vmap(
            lambda pos: target_position_to_log_angular_coeffs(
                pos, target_position_inverse_temperature, lmax
            )
        )(target_positions)
        true_radial_logits = models.safe_log(true_radial_weights)
        log_true_dist_coeffs = jax.vmap(
            models.compute_coefficients_of_logits_of_joint_distribution
        )(true_radial_logits, log_true_angular_coeffs)
        # We only support num_channels = 1.
        log_predicted_dist_coeffs = log_position_coeffs.reshape(
            log_true_dist_coeffs.shape
        )

        assert target_positions.shape == (num_graphs, 3)
        assert log_true_dist_coeffs.shape == (
            num_graphs,
            num_radii,
            log_true_dist_coeffs.irreps.dim,
        ), (
            log_true_dist_coeffs.shape,
            (num_graphs, num_radii, log_predicted_dist_coeffs.irreps.dim),
        )
        assert log_predicted_dist_coeffs.shape == (
            num_graphs,
            num_radii,
            log_predicted_dist_coeffs.irreps.dim,
        ), (
            log_predicted_dist_coeffs.shape,
            (num_graphs, num_radii, log_predicted_dist_coeffs.irreps.dim),
        )

        loss_position = jax.vmap(l2_loss_on_spheres)(
            log_true_dist_coeffs, log_predicted_dist_coeffs
        )
        assert loss_position.shape == (num_graphs,)

        return loss_position

    def factorized_position_loss() -> jnp.ndarray:
        """Computes the loss over position probabilities using separate losses for the radial and the angular components."""
        # Radial loss is simply the negative log-likelihood loss.
        loss_radial = -preds.globals.radial_logits.sum(axis=-1)

        # The angular loss is the KL divergence between the predicted and the true angular distributions.
        predicted_angular_logits = preds.globals.angular_logits
        res_beta, res_alpha = (
            predicted_angular_logits.res_beta,
            predicted_angular_logits.res_alpha,
        )

        target_positions = graphs.globals.target_positions
        log_true_angular_coeffs = jax.vmap(
            lambda pos: target_position_to_log_angular_coeffs(
                pos, target_position_inverse_temperature, lmax
            )
        )(target_positions)
        compute_grid_of_joint_distribution_fn = functools.partial(
            models.compute_grid_of_joint_distribution,
            res_beta=res_beta,
            res_alpha=res_alpha,
            quadrature=predicted_angular_logits.quadrature,
        )
        true_angular_dist = jax.vmap(
            compute_grid_of_joint_distribution_fn,
        )(jnp.ones((num_graphs, 1)), log_true_angular_coeffs)

        assert predicted_angular_logits.shape == (
            num_graphs,
            1,
            res_beta,
            res_alpha,
        ), predicted_angular_logits.shape

        loss_angular = jax.vmap(kl_divergence_on_spheres)(
            true_angular_dist, predicted_angular_logits
        )

        loss_position = loss_radial + loss_angular
        assert loss_position.shape == (num_graphs,)

        return loss_position

    def position_loss() -> jnp.ndarray:
        """Computes the loss over position probabilities."""
        if position_loss_type == "kl_divergence":
            return position_loss_with_kl_divergence()
        elif position_loss_type == "l2":
            return position_loss_with_l2()
        elif position_loss_type.startswith("factorized"):
            return factorized_position_loss()
        else:
            raise ValueError(f"Unsupported position loss type: {position_loss_type}.")

    # If we should predict a STOP for this fragment, we do not have to predict a position.
    loss_focus_and_atom_type = focus_and_atom_type_loss()
    loss_position = (1 - graphs.globals.stop) * position_loss()

    # Ignore position loss for graphs with less than, or equal to 3 atoms?
    # This is because there are symmetry-based degeneracies in the target distribution for these graphs.
    if ignore_position_loss_for_small_fragments:
        loss_position = jnp.where(n_node <= 3, 0, loss_position)

    total_loss = loss_focus_and_atom_type + loss_position
    return total_loss, (loss_focus_and_atom_type, loss_position)


@jax.profiler.annotate_function
def denoising_loss(
    preds: jnp.ndarray,
    graphs: datatypes.Fragments,
    position_noise: jnp.ndarray,
    center_at_zero: bool = True,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray]]:
    """Computes the loss for denoising atom positions."""
    num_graphs = graphs.n_node.shape[0]
    num_nodes = graphs.nodes.positions.shape[0]
    segment_ids = models.get_segment_ids(graphs.n_node, num_nodes)

    if position_noise is None or preds is None:
        return jnp.zeros((num_graphs,))

    # Subtract out the mean position noise.
    # This handles translation invariance.
    if center_at_zero:
        preds -= jraph.segment_mean(preds, segment_ids, num_graphs)[segment_ids]
        position_noise -= jraph.segment_mean(position_noise, segment_ids, num_graphs)[
            segment_ids
        ]

    # Compute the L2 loss.
    loss_denoising = jraph.segment_mean(
        jnp.sum(jnp.square(preds - position_noise), axis=-1),
        segment_ids,
        num_graphs,
    )
    assert loss_denoising.shape == (num_graphs,)

    # We predict a denoising loss only for finished fragments.
    loss_denoising = (graphs.globals.stop) * loss_denoising
    return loss_denoising, (loss_denoising, None)
