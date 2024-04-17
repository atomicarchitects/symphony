"""State for keeping track of training progress."""

from typing import Any, Dict, Optional, Callable
from clu import metrics
import flax
from flax.training import train_state
import chex
import optax

from symphony import datatypes


class TrainState(train_state.TrainState):
    """State for keeping track of training progress."""

    eval_apply_fn: Callable[
        [optax.Params, chex.PRNGKey, datatypes.Fragments], datatypes.Predictions
    ] = flax.struct.field(pytree_node=False)
    best_params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)
    step_for_best_params: int
    metrics_for_best_params: Optional[
        Dict[str, metrics.Collection]
    ] = flax.struct.field(pytree_node=True)
    train_metrics: metrics.Collection = flax.struct.field(pytree_node=True)

    def get_step(self) -> int:
        return int(self.step)
