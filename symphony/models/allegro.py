from typing import Callable, Optional
import jax
import jax.numpy as jnp
import allegro_jax
import haiku as hk
import e3nn_jax as e3nn

from symphony import datatypes


class Allegro(hk.Module):
    """Wrapper class for Allegro."""

    def __init__(
        self,
        num_species: int,
        r_max: float,
        avg_num_neighbors: float,
        max_ell: int,
        output_irreps: str,
        num_interactions: int,
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray],
        mlp_n_hidden: int,
        mlp_n_layers: int,
        n_radial_basis: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_species = num_species
        self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.max_ell = max_ell
        self.output_irreps = output_irreps
        self.num_interactions = num_interactions
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.n_radial_basis = n_radial_basis

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

        node_feats = jax.nn.one_hot(graphs.nodes.species, self.num_species)
        edge_feats = allegro_jax.Allegro(
            avg_num_neighbors=self.avg_num_neighbors,
            max_ell=self.max_ell,
            irreps=self.output_irreps,
            mlp_activation=self.mlp_activation,
            mlp_n_hidden=self.mlp_n_hidden,
            mlp_n_layers=self.mlp_n_layers,
            n_radial_basis=self.n_radial_basis,
            radial_cutoff=self.r_max,
            output_irreps=self.output_irreps,
            num_layers=self.num_interactions,
        )(node_feats, relative_positions, graphs.senders, graphs.receivers)
        
        # Aggregate edge features to nodes
        node_feats = jax.ops.segment_sum(edge_feats, graphs.receivers) + jax.ops.segment_sum(edge_feats, graphs.senders)
        return node_feats
