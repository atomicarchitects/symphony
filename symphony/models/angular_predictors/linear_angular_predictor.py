from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import chex
import e3nn_jax as e3nn


from symphony.models.angular_predictors import AngularPredictor


class LinearAngularPredictor(AngularPredictor):
    """A simple angular predictor, that applies a Linear layer to the conditioning."""

    def __init__(
        self,
        max_ell: int,
        num_channels: int,
        radial_mlp_num_layers: int,
        radial_mlp_latent_size: int,
        max_radius: int,
        res_beta: float,
        res_alpha: float,
        quadrature: str,
        sampling_inverse_temperature_factor: float,
        sampling_num_steps: int,
        sampling_init_step_size: float,
    ):
        super().__init__()
        self.max_ell = max_ell
        self.num_channels = num_channels
        self.radial_mlp_num_layers = radial_mlp_num_layers
        self.radial_mlp_latent_size = radial_mlp_latent_size
        self.max_radius = max_radius
        self.res_beta = res_beta
        self.res_alpha = res_alpha
        self.quadrature = quadrature
        self.sampling_inverse_temperature_factor = sampling_inverse_temperature_factor
        self.sampling_num_steps = sampling_num_steps
        self.sampling_init_step_size = sampling_init_step_size

    def coeffs(self, radius: float, conditioning: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        """Computes the spherical harmonic coefficients at the given radius."""
        radial_embed = e3nn.bessel(
            radius, self.radial_mlp_latent_size, x_max=self.max_radius
        )
        radial_embed = jnp.atleast_2d(radial_embed)
        radial_embed = e3nn.haiku.MultiLayerPerceptron(
            [self.radial_mlp_latent_size] * (self.radial_mlp_num_layers - 1)
            + [conditioning.irreps.num_irreps],
            act=jax.nn.swish,
            output_activation=True,
        )(radial_embed)

        conditioning *= radial_embed
        coeffs = e3nn.haiku.Linear(
            irreps_out=e3nn.s2_irreps(self.max_ell), channel_out=self.num_channels
        )(conditioning)
        assert coeffs.shape == (self.num_channels, (self.max_ell + 1) ** 2)

        return coeffs

    def unnormalized_logits(
        self, normalized_position: e3nn.IrrepsArray, coeffs: e3nn.IrrepsArray
    ) -> float:
        """Computes the unnormalized logits for the given position."""
        vals = e3nn.to_s2point(coeffs, normalized_position)
        vals = vals.array.squeeze(-1)
        assert vals.shape == (self.num_channels,), vals.shape

        logits = jax.scipy.special.logsumexp(vals, axis=-1)
        assert logits.shape == (), logits.shape
        return logits

    def log_prob(
        self, position: e3nn.IrrepsArray, conditioning: e3nn.IrrepsArray
    ) -> float:
        """Computes the logits for the given position and coefficients."""
        # Normalize the position.
        normalized_position = position / jnp.linalg.norm(position.array)
        assert normalized_position.shape == (3,), normalized_position.shape

        # Compute the coefficients at this radius.
        coeffs = self.coeffs(jnp.linalg.norm(position.array), conditioning)

        # We have to compute the log partition function, because the distribution is not normalized.
        prob_signal = e3nn.to_s2grid(
            coeffs,
            res_beta=self.res_beta,
            res_alpha=self.res_alpha,
            quadrature=self.quadrature,
        )
        assert prob_signal.shape == (
            self.num_channels,
            self.res_beta,
            self.res_alpha,
        )
        factor = jnp.max(prob_signal.grid_values)
        prob_signal = prob_signal.replace_values(prob_signal.grid_values - factor)
        prob_signal = prob_signal.replace_values(jnp.exp(prob_signal.grid_values))
        prob_signal = prob_signal.replace_values(
            jnp.sum(prob_signal.grid_values, axis=-3)
        )
        assert prob_signal.shape == (self.res_beta, self.res_alpha), prob_signal.shape

        log_Z = jnp.log(prob_signal.integrate().array.sum())
        assert log_Z.shape == (), log_Z.shape

        # We can compute the logits.
        vals = e3nn.to_s2point(coeffs, normalized_position)
        vals -= factor
        vals = vals.array.squeeze(-1)
        assert vals.shape == (self.num_channels,), vals.shape

        logits = jax.scipy.special.logsumexp(vals, axis=-1)
        assert logits.shape == (), logits.shape

        # jax.debug.print("grid_values_min={x}, grid_values_max={y}", x=jnp.min(prob_signal.grid_values), y=jnp.max(prob_signal.grid_values))
        # jax.debug.print("integral={x}", x=prob_signal.integrate())
        # jax.debug.print("normalized_position={x}", x=normalized_position)
        # jax.debug.print("coeffs={x}", x=coeffs)
        # jax.debug.print("vals={x}", x=vals)
        # jax.debug.print("logits={x}", x=logits)

        return logits - log_Z

    @staticmethod
    def coeffs_to_probability_distribution(
        coeffs: e3nn.IrrepsArray, res_beta: int, res_alpha: int, quadrature: str
    ) -> e3nn.SphericalSignal:
        """Converts the coefficients at this radius to a probability distribution."""
        num_channels = coeffs.shape[-2]

        prob_signal = e3nn.to_s2grid(
            coeffs, res_beta=res_beta, res_alpha=res_alpha, quadrature=quadrature
        )
        assert prob_signal.shape == (
            num_channels,
            res_beta,
            res_alpha,
        )

        prob_signal = prob_signal.replace_values(
            prob_signal.grid_values - jnp.max(prob_signal.grid_values)
        )
        prob_signal = prob_signal.replace_values(jnp.exp(prob_signal.grid_values))
        prob_signal = prob_signal.replace_values(
            jnp.sum(prob_signal.grid_values, axis=-3)
        )
        prob_signal /= prob_signal.integrate().array.sum()
        assert prob_signal.shape == (
            res_beta,
            res_alpha,
        )
        return prob_signal

    def sample(
        self, radius: float, conditioning: e3nn.IrrepsArray, inverse_temperature: float
    ) -> e3nn.IrrepsArray:
        """Samples from the learned distribution using the discretized grid."""
        # Compute the coefficients at this radius.
        coeffs = self.coeffs(radius, conditioning)

        # Scale coefficients by the inverse temperature.
        beta = self.sampling_inverse_temperature_factor * inverse_temperature
        coeffs *= beta

        # We have to compute the log partition function, because the distribution is not normalized.
        prob_signal = self.coeffs_to_probability_distribution(
            coeffs, self.res_beta, self.res_alpha, self.quadrature
        )

        # Sample from the distribution.
        key = hk.next_rng_key()
        key, sample_key = jax.random.split(key)
        beta_index, alpha_index = prob_signal.sample(sample_key)
        sample = prob_signal.grid_vectors[beta_index, alpha_index]
        assert sample.shape == (3,), sample.shape

        # Scale by the radius.
        return sample * radius

    def langevin_sample(
        self, radius: float, conditioning: e3nn.IrrepsArray, inverse_temperature: float
    ) -> e3nn.IrrepsArray:
        """Samples from the learned distribution using Langevin dynamics."""

        # Compute the coefficients at this radius.
        coeffs = self.coeffs(radius, conditioning)

        # Define the sampling inverse temperature factor.
        beta = self.sampling_inverse_temperature_factor * inverse_temperature

        def sample_from_uniform_distribution_on_sphere(
            key: chex.PRNGKey,
        ) -> e3nn.IrrepsArray:
            """Samples from a uniform distribution on the unit sphere."""
            z = jax.random.uniform(key, (3,), minval=-1, maxval=1)
            z /= jnp.linalg.norm(z)
            z = e3nn.IrrepsArray("1o", z)
            return z

        def score(sample: e3nn.IrrepsArray) -> float:
            """Computes the score at the given sample."""
            return jax.grad(self.unnormalized_logits, argnums=0)(sample, coeffs)

        def project_update_on_tangent_space(
            sample: e3nn.IrrepsArray, update: e3nn.IrrepsArray
        ) -> e3nn.IrrepsArray:
            """Projects the update on the tangent space of the sphere at the given sample."""
            return update - e3nn.dot(sample, update) * sample

        def apply_exponential_map(
            sample: e3nn.IrrepsArray, update: e3nn.IrrepsArray
        ) -> e3nn.IrrepsArray:
            """Applies the exponential map to the given sample and update."""
            update_norm = jnp.linalg.norm(update.array)
            return (
                jnp.cos(update_norm) * sample
                + jnp.sin(update_norm) * update / update_norm
            )

        def update(state: Tuple[e3nn.IrrepsArray, float], key: chex.PRNGKey):
            """Performs a single update of the state using Langevin dynamics."""
            sample, step_size = state

            # Compute Langevin dynamics update.
            key, noise_key = jax.random.split(key)
            update = step_size * score(sample)
            update += jnp.sqrt(2 * step_size / beta) * e3nn.normal("1o", noise_key)
            update = project_update_on_tangent_space(sample, update)
            new_sample = apply_exponential_map(sample, update)

            # Apply Metropolis-Hastings correction.
            key, mh_key = jax.random.split(key)
            log_acceptance_ratio = self.unnormalized_logits(
                new_sample, coeffs
            ) - self.unnormalized_logits(sample, coeffs)
            log_acceptance_ratio = jnp.minimum(0, log_acceptance_ratio)
            acceptance_ratio = jnp.exp(log_acceptance_ratio)
            acceptance = jax.random.bernoulli(mh_key, acceptance_ratio)
            new_sample = jnp.where(acceptance, new_sample.array, sample.array)
            new_sample = e3nn.IrrepsArray("1o", new_sample)

            new_step_size = step_size * (1 - 1 / (self.sampling_num_steps))
            return (new_sample, new_step_size), new_sample

        rng = hk.next_rng_key()
        rng, sample_rng = jax.random.split(rng)
        init_sample = sample_from_uniform_distribution_on_sphere(sample_rng)

        # Run Langevin dynamics on the unit sphere.
        (positions, _), _ = jax.lax.scan(
            update,
            (init_sample, self.sampling_init_step_size),
            jax.random.split(sample_rng, self.sampling_num_steps),
        )

        # Scale by the radius.
        return positions.array * radius
