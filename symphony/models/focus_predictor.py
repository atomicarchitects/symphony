from typing import Callable, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp


class FocusAndTargetSpeciesPredictor(hk.Module):
    """Predicts the focus and target species distribution over all nodes."""

    def __init__(
        self,
        latent_size: int,
        num_layers: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        num_species: int,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.activation = activation
        self.num_species = num_species

    def __call__(self, node_embeddings: e3nn.IrrepsArray) -> jnp.ndarray:
        num_nodes, _ = node_embeddings.shape
        node_embeddings = node_embeddings.filter(keep="0e")
        species_logits = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[self.latent_size] * (self.num_layers - 1)
            + [self.num_species],
            act=self.activation,
            output_activation=False,
        )(node_embeddings).array
        assert species_logits.shape == (num_nodes, self.num_species)
        return species_logits

