import haiku as hk
import jax.numpy as jnp
import distrax
import e3nn_jax as e3nn


class RationalQuadraticSpline(hk.Module):
    """A rational quadratic spline flow."""

    def __init__(
        self, num_bins: int, range_min: float, range_max: float, num_layers: int, num_param_mlp_layers: int
    ):
        super().__init__()
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        self.num_layers = num_layers
        self.num_param_mlp_layers = num_param_mlp_layers


    def create_flow(self, conditioning: e3nn.IrrepsArray) -> distrax.Bijector:
        """Creates a flow with the given conditioning."""
        if not conditioning.irreps.is_scalar():
            raise ValueError("Conditioning for flow must be scalars only.")
        conditioning = conditioning.array

        layers = []
        for _ in range(self.num_layers):
            param_dims = self.num_bins * 3 + 1
            params = hk.nets.MLP(
                [param_dims] * self.num_param_mlp_layers,
                activate_final=False,
                w_init=hk.initializers.RandomNormal(1e-4),
                b_init=hk.initializers.RandomNormal(1e-4),
            )(conditioning)
            layer = distrax.RationalQuadraticSpline(
                params,
                self.range_min,
                self.range_max,
                boundary_slopes="unconstrained",
                min_bin_size=1e-2,
            )
            layers.append(layer)

        flow = distrax.Inverse(distrax.Chain(layers))
        return flow

    def create_distribution(
        self, conditioning: e3nn.IrrepsArray
    ) -> distrax.Distribution:
        """Creates a distribution by composing a base distribution with a flow."""
        flow = self.create_flow(conditioning)
        mean = (self.range_min + self.range_max) / 2
        std = (self.range_max - self.range_min) / 20
        base_distribution = distrax.Independent(
            distrax.ClippedNormal(
                mean, std, minimum=self.range_min, maximum=self.range_max
            ),
            reinterpreted_batch_ndims=0,
        )
        dist = distrax.Transformed(base_distribution, flow)
        return dist

    def forward(
        self, base_samples: jnp.ndarray, conditioning: e3nn.IrrepsArray
    ) -> jnp.ndarray:
        """Applies the flow to the given samples from the base distribution."""
        flow = self.create_flow(conditioning)
        return flow.forward(base_samples)

    def log_prob(
        self, samples: jnp.ndarray, conditioning: e3nn.IrrepsArray
    ) -> jnp.ndarray:
        """Computes the log probability of the given samples."""
        assert conditioning.shape[:-1] == samples.shape[:-1], (
            conditioning.shape,
            samples.shape,
        )
        dist = self.create_distribution(conditioning)
        return dist.log_prob(samples)

    def sample(self, conditioning: e3nn.IrrepsArray) -> jnp.ndarray:
        """Samples from the learned distribution."""
        dist = self.create_distribution(conditioning)
        rng = hk.next_rng_key()
        return dist.sample(seed=rng, sample_shape=conditioning.shape[:-1])
