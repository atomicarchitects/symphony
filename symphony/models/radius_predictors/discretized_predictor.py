import haiku as hk
import jax.numpy as jnp
import e3nn_jax as e3nn
import distrax


from symphony.models.radius_predictors import RadiusPredictor


class DiscretizedRadialPredictor(RadiusPredictor):
    """A discrete distribution for radii."""

    def __init__(
        self,
        num_bins: int,
        range_min: float,
        range_max: float,
        num_layers: int,
        latent_size: int,
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
        conditioning = conditioning.filter("0e")

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
        dist = distrax.Categorical(logits=logits)
        return dist

    def log_prob(
        self, samples: e3nn.IrrepsArray, conditioning: e3nn.IrrepsArray
    ) -> jnp.ndarray:
        """Computes the log probability of the given samples."""
        radii = jnp.linalg.norm(samples.array, axis=-1)
        indices = jnp.argmin(jnp.abs(radii - self.radii()))
        dist = self.create_distribution(conditioning)
        return dist.log_prob(indices)

    def sample(self, conditioning: e3nn.IrrepsArray) -> jnp.ndarray:
        """Samples from the learned distribution."""
        dist = self.create_distribution(conditioning)
        rng = hk.next_rng_key()
        indices = dist.sample(seed=rng, sample_shape=conditioning.shape[:-1])
        return self.radii()[indices]
