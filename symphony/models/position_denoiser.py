from typing import Optional

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp

from symphony import datatypes


class PositionDenoiser(hk.Module):
    """Performs a one-step update to all atom positions."""

    def __init__(
        self,
        node_embedder: hk.Module,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.node_embedder = node_embedder

    def compute_node_embeddings(self, graphs: datatypes.Fragments) -> e3nn.IrrepsArray:
        """Computes the node embeddings for the target positions."""
        return self.node_embedder(graphs)

    def __call__(self, graphs: datatypes.Fragments) -> jnp.ndarray:
        # Project each embedding to a vector, representing the noise in input position.
        node_embeddings = self.compute_node_embeddings(graphs)
        position_noise = e3nn.haiku.Linear("1o", force_irreps_out=True)(node_embeddings)
        return position_noise.array
