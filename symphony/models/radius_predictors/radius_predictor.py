import abc
import haiku as hk
import e3nn_jax as e3nn


class RadiusPredictor(hk.Module, abc.ABC):
    """An abstract class for radius predictors."""

    @abc.abstractmethod
    def log_prob(self, samples: e3nn.IrrepsArray, conditioning: e3nn.IrrepsArray):
        """Computes the log probability of the given samples."""
        pass

    @abc.abstractmethod
    def sample(self, conditioning: e3nn.IrrepsArray):
        """Samples from the learned distribution."""
        pass
