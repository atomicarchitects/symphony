import haiku as hk
import abc

import e3nn_jax as e3nn


class AngularPredictor(hk.Module, abc.ABC):
    """An abstract class for angular predictors."""

    @abc.abstractmethod
    def log_prob(
        self, position: e3nn.IrrepsArray, conditioning: e3nn.IrrepsArray
    ) -> float:
        """Computes the log probability of the given position."""
        pass

    @abc.abstractmethod
    def sample(
        self, radius: float, conditioning: e3nn.IrrepsArray, inverse_temperature: float
    ) -> e3nn.IrrepsArray:
        """Samples from the learned distribution."""
        pass
