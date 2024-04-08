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
    radial_logits = -(
        (radii - jnp.tile(jnp.linalg.norm(target_position, axis=-1), radii.shape[0]))
        ** 2
    )
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
    radial_loss_scaling_factor: Optional[float],
    mask_atom_types: bool,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Computes the loss for the generation task.
    Args:
        preds (datatypes.Predictions): the model predictions
        graphs (datatypes.Fragment): a batch of graphs representing the current molecules
    """
    radial_bins = preds.nodes.radial_bins[0]  # Assume all radii are the same.
    num_radii = radial_bins.shape[0]
    num_graphs = graphs.n_node.shape[0]
    num_nodes = graphs.nodes.positions.shape[0]
    num_nodes_for_multifocus = graphs.nodes.target_positions.shape[1]
    n_node = graphs.n_node
    max_targets_per_graph = graphs.nodes.target_positions.shape[1]
    segment_ids = models.get_segment_ids(n_node, num_nodes)
    segment_ids_multifocus = jnp.repeat(jnp.arange(num_graphs), num_nodes_for_multifocus, axis=0)
    if target_position_lmax is None:
        lmax = preds.nodes.log_position_coeffs.irreps.lmax
    else:
        lmax = target_position_lmax

    def focus_and_atom_type_loss() -> jnp.ndarray:
        """Computes the loss over focus and atom types for all nodes."""
        species_logits = preds.nodes.focus_and_target_species_logits
        species_targets = graphs.nodes.focus_and_target_species_probs
        stop_logits = preds.globals.stop_logits
        stop_targets = graphs.globals.stop.astype(jnp.float32)

        # print(f"shape: {species_logits.shape}")
        # print(f"shape: {species_targets.shape}")
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

        position_logits = preds.nodes.position_logits
        res_beta, res_alpha, quadrature = (
            position_logits.res_beta,
            position_logits.res_alpha,
            position_logits.quadrature,
        )

        target_positions = graphs.nodes.target_positions
        target_position_mask = graphs.nodes.target_position_mask
        true_radial_weights = jax.vmap(
            jax.vmap(
                lambda pos: target_position_to_radius_weights(
                    pos, radius_rbf_variance, radial_bins
                )
            )
        )(target_positions)
        log_true_angular_coeffs = jax.vmap(
            jax.vmap(
                lambda pos: target_position_to_log_angular_coeffs(
                    pos, target_position_inverse_temperature, lmax
                )
            )
        )(target_positions)

        assert true_radial_weights.shape == (
            num_nodes,
            max_targets_per_graph,
            num_radii,
        )
        assert log_true_angular_coeffs.shape == (
            num_nodes,
            max_targets_per_graph,
            log_true_angular_coeffs.irreps.dim,
        ), log_true_angular_coeffs.shape

        compute_joint_distribution_fn = functools.partial(
            models.compute_grid_of_joint_distribution,
            res_beta=res_beta,
            res_alpha=res_alpha,
            quadrature=quadrature,
        )
        true_dist = jax.vmap(jax.vmap(compute_joint_distribution_fn))(
            true_radial_weights, log_true_angular_coeffs
        )

        assert true_dist.shape == (
            num_nodes,
            max_targets_per_graph,
            num_radii,
            res_beta,
            res_alpha,
        )
        assert target_position_mask.shape == (num_nodes, max_targets_per_graph)

        target_position_mask = target_position_mask[:, :, None, None, None]
        num_valid_targets = jnp.sum(target_position_mask, axis=1, keepdims=False)
        num_valid_targets = jnp.maximum(num_valid_targets, 1.0)
        mean_true_dist = jnp.sum(true_dist.grid_values * target_position_mask, axis=1)
        mean_true_dist = mean_true_dist / num_valid_targets
        assert mean_true_dist.shape == (num_nodes, num_radii, res_beta, res_alpha), (
            mean_true_dist.shape,
            (num_nodes, num_radii, res_beta, res_alpha),
        )

        mean_true_dist = e3nn.SphericalSignal(
            grid_values=mean_true_dist, quadrature=true_dist.quadrature
        )
        assert mean_true_dist.shape == (num_nodes, num_radii, res_beta, res_alpha), (
            mean_true_dist.shape,
            (num_nodes, num_radii, res_beta, res_alpha),
        )

        log_predicted_dist = position_logits

        assert log_predicted_dist.grid_values.shape == (
            num_nodes,
            num_radii,
            res_beta,
            res_alpha,
        )

        loss_position = jax.vmap(kl_divergence_on_spheres)(
            mean_true_dist, log_predicted_dist
        )

        loss_position = jraph.segment_sum(
            loss_position.reshape(-1, 1) * graphs.nodes.target_position_mask, segment_ids, num_graphs
        ).sum(axis=-1)

        assert loss_position.shape == (num_graphs,), print(loss_position.shape)

        return loss_position

    def position_loss_with_l2() -> jnp.ndarray:
        """Computes the loss over position probabilities using the L2 loss on the logits."""

        log_position_coeffs = preds.nodes.log_position_coeffs
        target_positions = graphs.nodes.target_positions
        target_position_mask = graphs.nodes.target_position_mask
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
        log_true_dist_coeffs.grid_values = log_true_dist_coeffs.grid_values[:, 0, :, :]
        mean_true_angular_coeffs = e3nn.SphericalSignal(
            grid_values=log_true_dist_coeffs.grid_values.mean(axis=0),
            quadrature=log_true_dist_coeffs.quadrature,
        )
        # We only support num_channels = 1.
        log_predicted_dist_coeffs = log_position_coeffs.reshape(
            log_true_dist_coeffs.shape
        )

        assert (target_positions.shape[0], target_positions.shape[-1]) == (
            num_graphs,
            3,
        )
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
            mean_true_angular_coeffs, log_predicted_dist_coeffs
        )
        assert loss_position.shape == (num_graphs,)

        return loss_position

    def factorized_position_loss(position_loss_type: str) -> jnp.ndarray:
        """Computes the loss over position probabilities using separate losses for the radial and the angular components."""

        position_logits = preds.nodes.position_logits
        res_beta, res_alpha, quadrature = (
            position_logits.res_beta,
            position_logits.res_alpha,
            position_logits.quadrature,
        )

        target_positions = graphs.nodes.target_positions
        target_position_mask = graphs.nodes.target_position_mask
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
        true_angular_dist = jax.vmap(
            compute_joint_distribution_fn,
        )(
            jnp.ones(
                (
                    num_graphs,
                    1,
                ),
            ),
            log_true_angular_coeffs,
        )  # (num_graphs, max_targets_per_graph, 1, res_beta, res_alpha)
        true_angular_dist.grid_values = true_angular_dist.grid_values[:, :, 0, :, :]
        target_position_mask_reshaped = target_position_mask.reshape(
            num_graphs, -1, 1, 1
        )
        num_target_positions = jnp.sum(
            target_position_mask_reshaped, axis=1, keepdims=True
        )
        num_target_positions = jnp.maximum(num_target_positions, 1.0)
        angular_dist_sum = jnp.sum(
            true_angular_dist.grid_values * target_position_mask_reshaped, axis=1
        )[:, None, :, :]
        angular_dist_mean = angular_dist_sum / num_target_positions
        mean_true_angular_dist = e3nn.SphericalSignal(
            grid_values=angular_dist_mean, quadrature=true_angular_dist.quadrature
        )

        sum_radial_weights = jnp.sum(
            true_radial_weights.reshape(num_graphs, -1, num_radii)
            * target_position_mask.reshape(num_graphs, -1, 1),
            axis=1,
        )
        mean_radial_weights = sum_radial_weights / num_target_positions.reshape(
            num_graphs, 1
        )
        radial_normalization = mean_radial_weights.sum(axis=-1, keepdims=True)
        mean_radial_weights /= jnp.where(
            radial_normalization == 0, 1.0, radial_normalization
        )
        assert mean_radial_weights.shape == (
            num_graphs,
            num_radii,
        ), mean_radial_weights.shape

        predicted_radial_logits = preds.nodes.radial_logits
        if predicted_radial_logits is None:
            position_logits = preds.nodes.position_logits
            position_probs = jax.vmap(models.position_logits_to_position_distribution)(
                position_logits
            )

            predicted_radial_dist = jax.vmap(
                models.position_distribution_to_radial_distribution,
            )(position_probs)
            predicted_radial_logits = models.safe_log(predicted_radial_dist)

        assert (
            predicted_radial_logits.shape == mean_radial_weights.shape
        ), predicted_radial_logits.shape

        if position_loss_type == "factorized_kl_divergence":
            loss_radial = jax.vmap(kl_divergence_for_radii)(
                mean_radial_weights,
                predicted_radial_logits,
            )
        elif position_loss_type == "factorized_earth_mover":
            loss_radial = jax.vmap(earthmover_distance_for_radii, in_axes=(0, 0, None))(
                mean_radial_weights, predicted_radial_logits, radial_bins
            )
        loss_radial = radial_loss_scaling_factor * loss_radial

        predicted_angular_logits = preds.nodes.angular_logits
        if predicted_angular_logits is None:
            position_probs = jax.vmap(models.position_logits_to_position_distribution)(
                position_logits
            )
            predicted_angular_dist = jax.vmap(
                models.position_distribution_to_angular_distribution,
            )(position_probs)
            predicted_angular_logits = predicted_angular_dist.apply(models.safe_log)

            # Add a dummy dimension for the radial bins.
            predicted_angular_logits.grid_values = predicted_angular_logits.grid_values[
                :, None, :, :
            ]

        assert predicted_angular_logits.shape == (
            num_graphs,
            1,
            res_beta,
            res_alpha,
        ), predicted_angular_logits.shape

        loss_angular = jax.vmap(kl_divergence_on_spheres)(
            mean_true_angular_dist, predicted_angular_logits
        )

        loss_position = loss_angular + loss_radial
        assert loss_position.shape == (num_graphs,)

        return loss_position

    def position_loss() -> jnp.ndarray:
        """Computes the loss over position probabilities."""
        if position_loss_type == "kl_divergence":
            return position_loss_with_kl_divergence()
        elif position_loss_type == "l2":
            return position_loss_with_l2()
        elif position_loss_type.startswith("factorized"):
            return factorized_position_loss(position_loss_type)
        else:
            raise ValueError(f"Unsupported position loss type: {position_loss_type}.")

    # If we should predict a STOP for this fragment, we do not have to predict a position.
    loss_focus_and_atom_type = focus_and_atom_type_loss()
    loss_position = (1 - graphs.globals.stop) * position_loss()

    # COMMENT LATER.
    # loss_position = jnp.zeros_like(loss_position)
    # loss_focus_and_atom_type = jnp.zeros_like(loss_focus_and_atom_type)

    # Mask out the loss for atom types?
    if mask_atom_types:
        loss_focus_and_atom_type = jnp.zeros_like(loss_focus_and_atom_type)

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


def clash_loss(
    graphs: datatypes.Fragments,
    atomic_numbers: jnp.ndarray,  # typically [1, 6, 7, 8, 9]
    atol: float = 0.0,
    rtol: float = 0.1,
) -> jnp.ndarray:
    """Hinge loss that penalizes clashes between atoms depending on their atomic numbers."""
    assert 0.0 <= rtol < 1.0
    assert 0.0 <= atol

    clash_dist = {
        (8, 1): 0.96,
        (1, 7): 1.00,
        (1, 6): 1.06,
        (7, 7): 1.10,
        (8, 6): 1.13,
        (6, 7): 1.15,
        (8, 7): 1.17,
        (6, 6): 1.20,
        (9, 6): 1.30,
        (1, 1): 1.51,
        (9, 1): 2.06,
        (8, 9): 2.14,
        (8, 8): 2.15,
        (9, 9): 2.15,
        (9, 7): 2.18,
    }
    clash = np.zeros((len(atomic_numbers), len(atomic_numbers)))
    for (zi, zj), dist in clash_dist.items():
        if np.isin(zi, atomic_numbers) and np.isin(zj, atomic_numbers):
            si = np.searchsorted(atomic_numbers, zi)
            sj = np.searchsorted(atomic_numbers, zj)
            clash[si, sj] = dist
            clash[sj, si] = dist

    i = graphs.senders
    j = graphs.receivers
    si = graphs.nodes.species[i]
    sj = graphs.nodes.species[j]
    ri = graphs.nodes.positions[i]
    rj = graphs.nodes.positions[j]
    rij = safe_norm(ri - rj, axis=-1)

    def hinge_loss(x):
        return jnp.maximum(0.0, x)

    return hinge_loss(clash[si, sj] * (1.0 - rtol) - atol - rij)
