import functools
from typing import Tuple

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import numpy as np

import datatypes
import models


def safe_log(x: jnp.ndarray) -> jnp.ndarray:
    """Computes the log of x, replacing 0 with 1 for numerical stability."""
    return jnp.log(jnp.where(x == 0, 1.0, x))


def safe_norm(x: jnp.ndarray, axis) -> jnp.ndarray:
    """Computes the norm of x, replacing 0 with 1 for numerical stability."""
    x2 = jnp.sum(x**2, axis=axis)
    return jnp.sqrt(jnp.where(x2 == 0, 1.0, x2))


@functools.partial(jax.profiler.annotate_function, name="generation_loss")
def generation_loss(
    preds: datatypes.Predictions,
    graphs: datatypes.Fragments,
    radius_rbf_variance: float,
    target_position_inverse_temperature: float,
    ignore_position_loss_for_small_fragments: bool,
    position_loss_type: str,
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
    n_node = graphs.n_node

    def atom_type_loss() -> jnp.ndarray:
        """Computes the loss over atom types."""
        logits = preds.nodes.target_species_logits
        target = graphs.nodes.target_species_probs
        # stop = graphs.globals.stop (= 1 - sum(target))

        assert logits.shape == target.shape == (num_nodes, num_elements)

        max = e3nn.scatter_max(jnp.max(logits, axis=1), nel=n_node, initial=0.0)
        max_ext = e3nn.scatter_max(
            jnp.max(logits, axis=1, keepdims=True),
            nel=n_node,
            map_back=True,
            initial=0.0,
        )
        assert max.shape == (num_graphs,)
        assert max_ext.shape == (num_nodes, 1)

        loss_atom_type = -(
            -max + e3nn.scatter_sum(jnp.sum(target * logits, axis=-1), nel=n_node)
        ) + jnp.log(
            jnp.exp(-max)
            + e3nn.scatter_sum(jnp.sum(jnp.exp(logits - max_ext), axis=1), nel=n_node)
        )
        assert loss_atom_type.shape == (num_graphs,)
        return loss_atom_type

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
        target_position_unit_vector = target_position / jnp.where(norm == 0, 1, norm)
        return target_position_inverse_temperature * e3nn.s2_dirac(
            target_position_unit_vector,
            lmax=preds.globals.position_coeffs.irreps.lmax,
            p_val=1,
            p_arg=-1,
        )

    def position_loss_with_kl_divergence() -> jnp.ndarray:
        """Computes the loss over position probabilities using the KL divergence."""

        position_logits = preds.globals.position_logits
        res_beta, res_alpha, quadrature = (
            position_logits.res_beta,
            position_logits.res_alpha,
            position_logits.quadrature,
        )

        def kl_divergence_on_spheres(
            true_radius_weights: jnp.ndarray,
            log_true_angular_coeffs: e3nn.IrrepsArray,
            log_predicted_dist: e3nn.SphericalSignal,
        ) -> jnp.ndarray:
            """Compute the KL divergence between two distributions on the spheres."""
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

        target_positions = graphs.globals.target_positions
        true_radius_weights = jax.vmap(target_position_to_radius_weights)(
            target_positions
        )
        log_true_angular_coeffs = jax.vmap(target_position_to_log_angular_coeffs)(
            target_positions
        )
        log_predicted_dist = position_logits

        assert true_radius_weights.shape == (num_graphs, num_radii)
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
            true_radius_weights, log_true_angular_coeffs, log_predicted_dist
        )
        assert loss_position.shape == (num_graphs,)
        return loss_position

    def position_loss_with_l2() -> jnp.ndarray:
        """Computes the loss over position probabilities using the L2 loss on the logits."""

        def l2_loss_on_spheres(
            true_radius_weights: jnp.ndarray,
            log_true_angular_coeffs: e3nn.IrrepsArray,
            log_predicted_dist_coeffs: e3nn.SphericalSignal,
        ):
            """Computes the L2 loss between the logits of two distributions on the spheres."""
            assert log_true_angular_coeffs.irreps == log_predicted_dist_coeffs.irreps

            log_true_radius_weights = jnp.log(true_radius_weights)
            log_true_radius_weights = e3nn.IrrepsArray(
                "0e", log_true_radius_weights[:, None]
            )

            log_true_angular_coeffs_tiled = jnp.tile(
                log_true_angular_coeffs.array, (num_radii, 1)
            )
            log_true_angular_coeffs_tiled = e3nn.IrrepsArray(
                log_true_angular_coeffs.irreps, log_true_angular_coeffs_tiled
            )

            log_true_dist_coeffs = e3nn.concatenate(
                [log_true_radius_weights, log_true_angular_coeffs_tiled], axis=1
            )
            log_true_dist_coeffs = e3nn.sum(log_true_dist_coeffs.regroup(), axis=-1)

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

        position_coeffs = preds.globals.position_coeffs
        target_positions = graphs.globals.target_positions
        true_radius_weights = jax.vmap(target_position_to_radius_weights)(
            target_positions
        )
        log_true_angular_coeffs = jax.vmap(target_position_to_log_angular_coeffs)(
            target_positions
        )
        log_predicted_dist_coeffs = position_coeffs

        assert target_positions.shape == (num_graphs, 3)
        assert true_radius_weights.shape == (num_graphs, num_radii)
        assert log_true_angular_coeffs.shape == (
            num_graphs,
            log_true_angular_coeffs.irreps.dim,
        )
        assert log_predicted_dist_coeffs.shape == (
            num_graphs,
            num_radii,
            log_predicted_dist_coeffs.irreps.dim,
        )

        loss_position = jax.vmap(l2_loss_on_spheres)(
            true_radius_weights, log_true_angular_coeffs, log_predicted_dist_coeffs
        )
        assert loss_position.shape == (num_graphs,)
        return loss_position

    def position_loss() -> jnp.ndarray:
        """Computes the loss over position probabilities."""
        if position_loss_type == "kl_divergence":
            return position_loss_with_kl_divergence()
        elif position_loss_type == "l2":
            return position_loss_with_l2()
        else:
            raise ValueError(f"Unsupported position loss type: {position_loss_type}.")

    # If this is the last step in the generation process, we do not have to predict atom type and position.
    loss_atom_type = atom_type_loss()
    loss_position = position_loss() * (1 - graphs.globals.stop)

    # Ignore position loss for graphs with less than, or equal to 3 atoms.
    # This is because there are symmetry-based degeneracies in the target distribution for these graphs.
    if ignore_position_loss_for_small_fragments:
        loss_position = jnp.where(graphs.n_node <= 3, 0, loss_position)

    total_loss = loss_atom_type + loss_position
    return total_loss, (loss_atom_type, loss_position)


def clash_loss(
    graphs: datatypes.Fragments,
    atomic_numbers: jnp.ndarray,  # typically [1, 6, 7, 8, 9]
    atol: float = 0.0,
    rtol: float = 0.1,
) -> jnp.ndarray:
    """Hinge loss that penalizes clashes between atoms depending on their atomic numbers."""
    assert rtol >= 0.0
    assert rtol < 1.0
    assert atol >= 0.0

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
