from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp


from symphony import datatypes


class FocusAndTargetSpeciesPredictor(hk.Module):
    """Predicts the focus and target species distribution over all nodes."""

    def __init__(
        self,
        node_embedder: hk.Module,
        latent_size: int,
        num_layers: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        num_species: int,
        global_embedder: Optional[hk.Module] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.node_embedder = node_embedder
        self.global_embedder = global_embedder
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.activation = activation
        self.num_species = num_species

    def compute_node_embeddings(self, graphs: datatypes.Fragments) -> e3nn.IrrepsArray:
        """Computes the node embeddings for the target positions."""
        node_embeddings = self.node_embedder(graphs)

        # Concatenate global embeddings to node embeddings.
        if self.global_embedder is not None:
            graphs_with_node_embeddings = graphs._replace(nodes=node_embeddings)
            global_embeddings = self.global_embedder(graphs_with_node_embeddings)
            node_embeddings = e3nn.concatenate(
                [node_embeddings, global_embeddings], axis=-1
            )

        return node_embeddings

    def __call__(
        self, graphs: datatypes.Fragments, inverse_temperature: float = 1.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        num_graphs = graphs.n_node.shape[0]

        # Get the node embeddings.
        node_embeddings = self.compute_node_embeddings(graphs)

        num_nodes, _ = node_embeddings.shape
        node_embeddings = node_embeddings.filter(keep="0e")
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
