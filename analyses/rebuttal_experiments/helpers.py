import e3nn_jax as e3nn
import chex
import jax
import jax.numpy as jnp
import functools

from symphony import models


def average_target_distributions(coeffs: e3nn.IrrepsArray, res_alpha: int, res_beta: int) -> e3nn.SphericalSignal:
    """Averages the target distributions over the batch dimension."""
    dist = jax.vmap(
        lambda coeff: models.compute_grid_of_joint_distribution(
            radial_weights=jnp.ones(
                1,
            ),
            quadrature="gausslegendre",
            log_angular_coeffs=coeff,
            res_beta=res_beta,
            res_alpha=res_alpha,
        )
    )(coeffs)
    dist.grid_values = dist.grid_values.mean(axis=0)
    return dist


def sample_from_angular_distribution(
    angular_probs: e3nn.SphericalSignal, rng: chex.PRNGKey
):
    """Sample a unit vector from an angular distribution."""
    beta_index, alpha_index = angular_probs.sample(rng)
    return angular_probs.grid_vectors[beta_index, alpha_index]


def sample_from_position_distribution(
    position_probs: e3nn.SphericalSignal, radii: jnp.ndarray, rng: chex.PRNGKey
) -> jnp.ndarray:
    """Samples a position vector from a distribution over all positions."""
    num_radii = radii.shape[0]
    assert radii.shape == (num_radii,)
    assert position_probs.shape == (
        num_radii,
        position_probs.res_beta,
        position_probs.res_alpha,
    )

    # Sample a radius.
    radial_probs = models.position_distribution_to_radial_distribution(position_probs)
    rng, radius_rng = jax.random.split(rng)
    radius_index = jax.random.choice(radius_rng, num_radii, p=radial_probs)

    # Get the angular probabilities.
    angular_probs = (
        position_probs[radius_index] / position_probs[radius_index].integrate()
    )

    # Sample angles.
    rng, angular_rng = jax.random.split(rng)
    unit_vector = sample_from_angular_distribution(angular_probs, angular_rng)

    # Combine the radius and angles to get the position vectors.
    position_vector = radii[radius_index] * unit_vector
    return position_vector


@functools.partial(jax.jit, static_argnames=("res_alpha", "res_beta"))
def coeffs_to_distribution(coeffs, res_alpha, res_beta):
    """Converts the coefficients to a distribution over positions."""
    log_predicted_dist = models.log_coeffs_to_logits(
        coeffs, res_beta=res_beta, res_alpha=res_alpha, num_radii=1
    )
    predicted_dist = models.position_logits_to_position_distribution(log_predicted_dist)
    return predicted_dist


def sample_from_dist(dist, rng, num_samples=100):
    """Samples from a distribution."""
    samples_rngs = jax.random.split(rng, num_samples)
    samples = jax.vmap(
        lambda key: sample_from_position_distribution(
            dist,
            jnp.ones(
                1,
            ),
            key,
        )
    )(samples_rngs)
    return samples


def closest_distances(points, target_points):
    """Returns the closest distances of points to target points."""
    distances = jnp.linalg.norm(points[:, None, :] - target_points[None, :, :], axis=-1)
    closest_indices = jnp.argmin(distances, axis=-1)
    sample_closest_distances = distances[jnp.arange(points.shape[0]), closest_indices]
    return sample_closest_distances


@functools.partial(jax.jit, static_argnames=("num_samples"))
def rmse_of_samples(dist, target_points, rng, num_samples=100):
    """Returns the mean and std of the closest distances of samples to target points."""
    samples = sample_from_dist(dist, rng, num_samples=num_samples)
    sample_closest_distances = closest_distances(samples, target_points)
    mean_dist = jnp.mean(sample_closest_distances)
    std_dist = jnp.std(sample_closest_distances)
    return mean_dist, std_dist
