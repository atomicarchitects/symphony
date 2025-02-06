from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
from e3nn_jax.experimental.point_convolution import MessagePassingConvolutionHaiku, radial_basis
import haiku as hk
import jax.numpy as jnp
import jax


from symphony import datatypes


class FocusAndTargetSpeciesPredictor(hk.Module):
    """Predicts the focus and target species distribution over all nodes."""

    def __init__(
        self,
        node_embedder_fn: Callable[[], hk.Module],
        latent_size: int,
        num_layers: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        num_species: int,
        k: int = -1,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.node_embedder_short = node_embedder_fn()
        self.node_embedder_long = node_embedder_fn()
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.activation = activation
        self.num_species = num_species
        self.k = k
        self.big_embedding_size = 384


    def node_embedder(self, graphs: datatypes.Fragments) -> e3nn.IrrepsArray:
        num_edges = graphs.senders.shape[0]

        # Get the node embeddings.
        short_edges_start = jnp.cumsum(
            jnp.concatenate([jnp.zeros(1), graphs.n_edge])
        )
        long_edges_start = short_edges_start[:-1] + graphs.globals.n_short_edge
        mask_short = jax.vmap(
            lambda i: (i >= short_edges_start[i]) & (i < long_edges_start[i])
        )(jnp.arange(num_edges)).squeeze()
        mask_long = jax.vmap(
            lambda i: (i >= long_edges_start[i]) & (i < short_edges_start[i + 1])
        )(jnp.arange(num_edges)).squeeze()
        graphs_short = graphs._replace(
            edges=graphs.edges,
            senders=graphs.senders[jnp.where(mask_short, size=num_edges)],
            receivers=graphs.receivers[jnp.where(mask_short, size=num_edges)],
        )
        graphs_long = graphs._replace(
            edges=graphs.edges,
            senders=graphs.senders[jnp.where(mask_long, size=num_edges)],
            receivers=graphs.receivers[jnp.where(mask_long, size=num_edges)],
        )
        node_embeddings_short = self.node_embedder_short(graphs_short)
        node_embeddings_long = self.node_embedder_long(graphs_long)

        return node_embeddings_short.filter(keep="0e") * node_embeddings_long.filter(keep="0e")

    def __call__(
        self, graphs: datatypes.Fragments, inverse_temperature: float = 1.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        num_graphs = graphs.n_node.shape[0]
        num_nodes = graphs.nodes.positions.shape[0]

        # Get the node embeddings.
        node_embeddings = self.node_embedder(graphs)
        focus_and_target_species_logits = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[self.latent_size] * (self.num_layers - 1)
            + [self.num_species],
            act=self.activation,
            output_activation=False,
        )(node_embeddings).array
        stop_logits = jnp.zeros((num_graphs,))

        assert focus_and_target_species_logits.shape == (num_nodes, self.num_species)
        assert stop_logits.shape == (num_graphs,)

        # Scale the logits by the inverse temperature.
        focus_and_target_species_logits *= inverse_temperature
        stop_logits *= inverse_temperature

        return focus_and_target_species_logits, stop_logits
