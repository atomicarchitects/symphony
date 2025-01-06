from typing import Callable, Optional
import jax.numpy as jnp
import nequip_jax
import haiku as hk
import e3nn_jax as e3nn

from symphony import datatypes


class NequIP(hk.Module):
    """Wrapper class for NequIP."""

    def __init__(
        self,
        num_species: int,
        r_max: float,
        avg_num_neighbors: float,
        max_ell: int,
        init_embedding_dims: int,
        hidden_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        num_interactions: int,
        even_activation: Callable[[jnp.ndarray], jnp.ndarray],
        odd_activation: Callable[[jnp.ndarray], jnp.ndarray],
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray],
        mlp_n_hidden: int,
        mlp_n_layers: int,
        n_radial_basis: int,
        skip_connection: bool,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_species = num_species
        self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.max_ell = max_ell
        self.init_embedding_dims = init_embedding_dims
        self.hidden_irreps = hidden_irreps
        self.output_irreps = output_irreps
        self.num_interactions = num_interactions
        self.even_activation = even_activation
        self.odd_activation = odd_activation
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.n_radial_basis = n_radial_basis
        self.skip_connection = skip_connection

    def __call__(
        self,
        graphs: datatypes.Fragments,
    ):
        relative_positions = (
            graphs.nodes.positions[graphs.receivers]
            - graphs.nodes.positions[graphs.senders]
        )
        relative_positions = relative_positions / self.r_max
        relative_positions = e3nn.IrrepsArray("1o", relative_positions)

        species = graphs.nodes.species
        node_feats = hk.Embed(self.num_species, self.init_embedding_dims)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)

        for interaction in range(self.num_interactions):
            new_node_feats = nequip_jax.NEQUIPESCNLayerHaiku(
                avg_num_neighbors=self.avg_num_neighbors,
                num_species=self.num_species,
                output_irreps=self.hidden_irreps,
                even_activation=self.even_activation,
                odd_activation=self.odd_activation,
                mlp_activation=self.mlp_activation,
                mlp_n_hidden=self.mlp_n_hidden,
                mlp_n_layers=self.mlp_n_layers,
                n_radial_basis=self.n_radial_basis,
            )(
                relative_positions, node_feats, species, graphs.senders, graphs.receivers
            )
            if interaction < self.num_interactions - 1:
                new_node_feats = e3nn.haiku.Linear(
                    self.hidden_irreps, force_irreps_out=True
                )(new_node_feats)
            else:
                new_node_feats = e3nn.haiku.Linear(
                    self.output_irreps, force_irreps_out=True
                )(new_node_feats)

            if self.skip_connection and interaction > 0:
                new_node_feats += node_feats
            node_feats = new_node_feats


        alpha = 0.5 ** jnp.array(node_feats.irreps.ls)
        node_feats = node_feats * alpha
        return node_feats
