
import haiku as hk
import jax.numpy as jnp
import e3nn_jax as e3nn
import distrax
import jax


class DiscretizedPredictor(hk.Module):
    """A learnable discrete distribution."""

    def __init__(
        self, num_bins: int, range_min: float, range_max: float, num_layers: int, latent_size: int
    ):
        super().__init__()
        self.num_bins = num_bins
        self.range_min = range_min
        self.range_max = range_max
        self.num_layers = num_layers
        self.latent_size = latent_size


    def radii(self) -> jnp.ndarray:
        return jnp.linspace(self.range_min, self.range_max, self.num_bins)

    def predict_logits(self, conditioning: e3nn.IrrepsArray) -> distrax.Bijector:
        """Predicts the logits."""
        if not conditioning.irreps.is_scalar():
            raise ValueError("Conditioning must be scalars only.")
        conditioning = conditioning.array

        logits = hk.nets.MLP(
            [self.latent_size] * (self.num_layers - 1) + [self.num_bins],
            activate_final=False,
        )(conditioning)
        return logits

    def create_distribution(
        self, conditioning: e3nn.IrrepsArray
    ) -> distrax.Distribution:
        """Creates a distribution."""
        logits = self.predict_logits(conditioning)
        # jax.debug.print("logits={x}", x=logits)
        dist = distrax.Categorical(logits=logits)
        return dist

    def log_prob(
        self, samples: jnp.ndarray, conditioning: e3nn.IrrepsArray
    ) -> jnp.ndarray:
        """Computes the log probability of the given samples."""
        assert conditioning.shape[:-1] == samples.shape[:-1], (
            conditioning.shape,
            samples.shape,
        )
        indices = jnp.argmin(
            jnp.abs(samples - self.radii())
        )
        # jax.debug.print("indices={x}", x=indices)
        dist = self.create_distribution(conditioning)
        # jax.debug.print("indices={x}", x=indices)
        # jax.debug.print("log_prob={x}", x=dist.log_prob(indices))
        # jax.debug.print("")
        # return jnp.ones_like(indices, dtype=jnp.float32)
        return dist.log_prob(indices)

    def sample(self, conditioning: e3nn.IrrepsArray) -> jnp.ndarray:
        """Samples from the learned distribution."""
        dist = self.create_distribution(conditioning)
        rng = hk.next_rng_key()
        indices = dist.sample(seed=rng, sample_shape=conditioning.shape[:-1])
        return self.radii()[indices]
