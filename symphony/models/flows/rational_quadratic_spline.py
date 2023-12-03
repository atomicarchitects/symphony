import haiku as hk
import jax
import jax.numpy as jnp
import distrax


class RationalQuadraticSpline(hk.Module):
    """A rational quadratic spline flow."""

    def __init__(
        self, num_bins: int, range_min: float, range_max: float, num_layers: int
    ):
        super().__init__()
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        self.num_layers = num_layers

    def create_flow(self, conditioning: jnp.ndarray) -> distrax.Bijector:
        """Creates a flow with the given conditioning."""
        layers = []
        for _ in range(self.num_layers):
            params = hk.nets.MLP([8, self.num_bins * 3 + 1], activate_final=False)(
                conditioning
            )
            layer = distrax.RationalQuadraticSpline(
                params, self.range_min, self.range_max
            )
            layers.append(layer)

        flow = distrax.Inverse(distrax.Chain(layers))
        return flow

    def create_distribution(self, conditioning: jnp.ndarray) -> distrax.Distribution:
        """Creates a distribution by composing a base distribution with a flow."""
        flow = self.create_flow(conditioning)
        base_distribution = distrax.Independent(
            distrax.Uniform(self.range_min, self.range_max), reinterpreted_batch_ndims=0
        )

        dist = distrax.Transformed(base_distribution, flow)
        return dist

    def forward(
        self, base_samples: jnp.ndarray, conditioning: jnp.ndarray
    ) -> jnp.ndarray:
        """Applies the flow to the given samples from the base distribution."""
        flow = self.create_flow(conditioning)
        return flow.forward(base_samples)

    def log_prob(self, samples: jnp.ndarray, conditioning: jnp.ndarray) -> jnp.ndarray:
        """Computes the log probability of the given samples."""
        assert conditioning.shape[:-1] == samples.shape[:-1], (
            conditioning.shape,
            samples.shape,
        )
        dist = self.create_distribution(conditioning)
        return dist.log_prob(samples)

    def sample(self, conditioning: jnp.ndarray) -> jnp.ndarray:
        """Samples from the learned distribution."""
        dist = self.create_distribution(conditioning)
        rng = hk.next_rng_key()
        return dist.sample(seed=rng, sample_shape=conditioning.shape[:-1])
