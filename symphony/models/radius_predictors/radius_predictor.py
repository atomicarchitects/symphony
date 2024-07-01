import abc
import haiku as hk
import e3nn_jax as e3nn
import distrax


class RadiusPredictor(hk.Module, abc.ABC):
    """An abstract class for radius predictors."""

    @abc.abstractmethod
    def log_prob(self, samples: e3nn.IrrepsArray, conditioning: e3nn.IrrepsArray):
        """Computes the log probability of the given samples."""

    @abc.abstractmethod
    def sample(self, conditioning: e3nn.IrrepsArray):
        """Samples from the learned distribution."""

    @abc.abstractmethod
    def create_distribution(
        self, conditioning: e3nn.IrrepsArray
    ) -> distrax.Distribution:
        """Creates a distribution."""

