from typing import Callable, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp

from symphony import datatypes


class PositionUpdater(hk.Module):
    """Performs a one-step update to all atom positions."""

    def __init__(
        self,
        node_embedder_fn: Callable[[], hk.Module],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.node_embedder = node_embedder_fn()

    def compute_node_embeddings(self, graphs: datatypes.Fragments) -> e3nn.IrrepsArray:
        """Computes the node embeddings for the target positions."""
        return self.node_embedder(graphs)

    def __call__(self, graphs: datatypes.Fragments) -> jnp.ndarray:
        # Project each embedding to a vector, representing the update in input positions.
        node_embeddings = self.compute_node_embeddings(graphs)
        position_update = e3nn.haiku.Linear("1o", force_irreps_out=True)(
            node_embeddings
        )
        return position_update.array
