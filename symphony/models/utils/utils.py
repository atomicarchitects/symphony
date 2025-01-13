"""Definition of the generative models."""

from typing import Tuple

import chex
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph


def get_atomic_numbers(
    species: jnp.ndarray, atomic_numbers: jnp.ndarray
) -> jnp.ndarray:
    """Returns the atomic numbers for the species."""
    return jnp.asarray(atomic_numbers)[species]


def get_first_node_indices(graphs: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the indices of the focus nodes in each graph."""
    return jnp.concatenate((jnp.asarray([0]), jnp.cumsum(graphs.n_node)[:-1]))


def segment_softmax_2D_with_stop(
    species_logits: jnp.ndarray,
    stop_logits: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the species probabilities and stop probabilities with segment softmax over 2D arrays of species logits."""
    # Subtract the max to avoid numerical issues.
    logits_max = jraph.segment_max(
        species_logits, segment_ids, num_segments=num_segments
    ).max(axis=-1)
    logits_max = jnp.maximum(logits_max, stop_logits)
    logits_max = jax.lax.stop_gradient(logits_max)
    species_logits -= logits_max[segment_ids, None]
    stop_logits -= logits_max

    # Normalize exp() by all nodes, all atom types, and the stop for each graph.
    exp_species_logits = jnp.exp(species_logits)
    exp_species_logits_summed = jnp.sum(exp_species_logits, axis=-1)
    normalizing_factors = jraph.segment_sum(
        exp_species_logits_summed, segment_ids, num_segments=num_segments
    )
    exp_stop_logits = jnp.exp(stop_logits)

    normalizing_factors += exp_stop_logits
    species_probs = exp_species_logits / normalizing_factors[segment_ids, None]
    stop_probs = exp_stop_logits / normalizing_factors

    return species_probs, stop_probs


def get_segment_ids(
    n_node: jnp.ndarray,
    num_nodes: int,
) -> jnp.ndarray:
    """Returns the segment ids for each node in the graphs."""
    num_graphs = n_node.shape[0]

    return jnp.repeat(
        jnp.arange(num_graphs), n_node, axis=0, total_repeat_length=num_nodes
    )


def segment_sample_2D(
    species_probabilities: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    num_targets: int,
    rng: chex.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample indices from a categorical distribution across each segment.
    Args:
        species_probabilities: A 2D array of probabilities.
        segment_ids: A 1D array of segment ids.
        num_segments: The number of segments.
        rng: A PRNG key.
    Returns:
        A 1D array of sampled indices, one for each segment.
    """
    num_nodes, num_species = species_probabilities.shape

    # Normalize the probabilities to sum up for 1 over all nodes in each graph.
    species_probabilities_summed = jraph.segment_sum(
        species_probabilities.sum(axis=-1), segment_ids, num_segments
    )
    species_probabilities = (
        species_probabilities / species_probabilities_summed[segment_ids, None]
    )

    def sample_for_segment(rng: chex.PRNGKey, segment_id: int) -> Tuple[float, float]:
        """Samples a node and species index for a single segment."""
        node_rng, logit_rng, rng = jax.random.split(rng, num=3)
        node_index = jax.random.choice(
            node_rng,
            jnp.arange(num_nodes),
            p=jnp.where(
                segment_id == segment_ids, species_probabilities.sum(axis=-1), 0.0
            ),
        )
        normalized_probs_for_index = species_probabilities[node_index] / jnp.sum(
            species_probabilities[node_index]
        )
        species_index = jax.random.choice(
            logit_rng,
            jnp.arange(num_species),
            shape=(num_targets,),
            replace=True,
            p=normalized_probs_for_index
        )
        return node_index, species_index

    rngs = jax.random.split(rng, num_segments)
    node_indices, species_indices = jax.vmap(sample_for_segment)(
        rngs, jnp.arange(num_segments)
    )
    assert node_indices.shape == (num_segments,)
    assert species_indices.shape == (num_segments, num_targets)
    return node_indices, species_indices


def log_coeffs_to_logits(
    log_coeffs: e3nn.IrrepsArray, res_beta: int, res_alpha: int, num_radii: int
) -> e3nn.SphericalSignal:
    """Converts coefficients of the logits to a SphericalSignal representing the logits."""
    num_channels = log_coeffs.shape[0]
    assert log_coeffs.shape == (
        num_channels,
        num_radii,
        log_coeffs.irreps.dim,
    ), f"{log_coeffs.shape}"

    log_dist = e3nn.to_s2grid(
        log_coeffs, res_beta, res_alpha, quadrature="gausslegendre", p_val=1, p_arg=-1
    )
    assert log_dist.shape == (num_channels, num_radii, res_beta, res_alpha)

    # Combine over all channels.
    log_dist.grid_values = jax.scipy.special.logsumexp(log_dist.grid_values, axis=0)
    assert log_dist.shape == (num_radii, res_beta, res_alpha)

    # Subtract the max to avoid numerical issues.
    max_logit = jnp.max(log_dist.grid_values)
    max_logit = jax.lax.stop_gradient(max_logit)
    log_dist.grid_values -= max_logit

    return log_dist


def position_logits_to_position_distribution(
    position_logits: e3nn.SphericalSignal,
) -> e3nn.SphericalSignal:
    """Converts logits to a SphericalSignal representing the position distribution."""

    assert len(position_logits.shape) == 3  # [num_radii, res_beta, res_alpha]
    max_logit = jnp.max(position_logits.grid_values)
    max_logit = jax.lax.stop_gradient(max_logit)

    position_probs = position_logits.apply(lambda logit: jnp.exp(logit - max_logit))

    position_probs.grid_values /= position_probs.integrate().array.sum()
    return position_probs


def safe_log(x: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """Computes the log of x, replacing 0 with a small value for numerical stability."""
    return jnp.log(jnp.where(x == 0, eps, x))


def position_distribution_to_radial_distribution(
    position_probs: e3nn.SphericalSignal,
) -> jnp.ndarray:
    """Computes the marginal radial distribution from a logits of a distribution over all positions."""
    assert len(position_probs.shape) == 3  # [num_radii, res_beta, res_alpha]
    return position_probs.integrate().array.squeeze(axis=-1)  # [..., num_radii]


def position_distribution_to_angular_distribution(
    position_probs: e3nn.SphericalSignal,
) -> jnp.ndarray:
    """Returns the marginal radial distribution for a logits of a distribution over all positions."""
    assert len(position_probs.shape) == 3  # [num_radii, res_beta, res_alpha]
    position_probs.grid_values = position_probs.grid_values.sum(axis=0)
    return position_probs


def compute_grid_of_joint_distribution(
    radial_weights: jnp.ndarray,
    log_angular_coeffs: e3nn.IrrepsArray,
    res_beta: int,
    res_alpha: int,
    quadrature: str,
) -> e3nn.SphericalSignal:
    """Combines radial weights and angular coefficients to get a distribution on the spheres."""
    # Convert coefficients to a distribution on the sphere.
    log_angular_dist = e3nn.to_s2grid(
        log_angular_coeffs,
        res_beta,
        res_alpha,
        quadrature=quadrature,
        p_val=1,
        p_arg=-1,
    )

    # Subtract the maximum value for numerical stability.
    log_angular_dist_max = jnp.max(
        log_angular_dist.grid_values, axis=(-2, -1), keepdims=True
    )
    log_angular_dist_max = jax.lax.stop_gradient(log_angular_dist_max)
    log_angular_dist = log_angular_dist.apply(lambda x: x - log_angular_dist_max)

    # Convert to a probability distribution, by taking the exponential and normalizing.
    angular_dist = log_angular_dist.apply(jnp.exp)
    angular_dist = angular_dist / angular_dist.integrate()

    # Check that shapes are correct.
    num_radii = radial_weights.shape[0]
    assert angular_dist.shape == (
        res_beta,
        res_alpha,
    )

    # Mix in the radius weights to get a distribution over all spheres.
    dist = radial_weights * angular_dist[None, :, :]
    assert dist.shape == (num_radii, res_beta, res_alpha)
    return dist


def compute_coefficients_of_logits_of_joint_distribution(
    radial_logits: jnp.ndarray,
    log_angular_coeffs: e3nn.IrrepsArray,
) -> e3nn.IrrepsArray:
    """Combines radial weights and angular coefficients to get a distribution on the spheres."""
    radial_logits = e3nn.IrrepsArray("0e", radial_logits[:, None])
    log_dist_coeffs = jax.vmap(
        lambda log_radial_weight: e3nn.concatenate(
            [log_radial_weight, log_angular_coeffs]
        )
    )(radial_logits)
    log_dist_coeffs = e3nn.sum(log_dist_coeffs.regroup(), axis=-1)

    num_radii = radial_logits.shape[0]
    assert log_dist_coeffs.shape == (num_radii, log_dist_coeffs.irreps.dim)

    return log_dist_coeffs
