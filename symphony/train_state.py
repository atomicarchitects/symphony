"""State for keeping track of training progress."""

from typing import Any, Dict, Optional
from clu import metrics
import flax
from flax.training import train_state


class TrainState(train_state.TrainState):
    """State for keeping track of training progress."""

    best_params: flax.core.FrozenDict[str, Any]
    step_for_best_params: float
    metrics_for_best_params: Optional[Dict[str, metrics.Collection]]
    train_metrics: metrics.Collection

    def get_step(self) -> int:
        return int(self.step[0])
