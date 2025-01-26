from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp


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
        self.node_embedder = node_embedder_fn()
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.activation = activation
        self.num_species = num_species
        self.k = k

    def __call__(
        self, graphs: datatypes.Fragments, inverse_temperature: float = 1.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        num_graphs = graphs.n_node.shape[0]

        # Get the node embeddings.
        node_embeddings = self.node_embedder(graphs)

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
