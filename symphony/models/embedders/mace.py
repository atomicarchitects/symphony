from typing import Optional
import mace_jax.modules
import haiku as hk
import e3nn_jax as e3nn

from symphony import datatypes


class MACE(hk.Module):
    """Wrapper class for MACE."""

    def __init__(
        self,
        output_irreps: str,
        hidden_irreps: str,
        readout_mlp_irreps: str,
        r_max: float,
        num_interactions: int,
        avg_num_neighbors: int,
        num_species: int,
        max_ell: int,
        num_basis_fns: int,
        soft_normalization: Optional[float],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.output_irreps = output_irreps
        self.r_max = r_max
        self.num_interactions = num_interactions
        self.hidden_irreps = hidden_irreps
        self.readout_mlp_irreps = readout_mlp_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.num_species = num_species
        self.max_ell = max_ell
        self.num_basis_fns = num_basis_fns
        self.soft_normalization = soft_normalization

    def __call__(self, graphs: datatypes.Fragments) -> e3nn.IrrepsArray:
        """Returns node embeddings for input graphs.
        Inputs:
            graphs: a jraph.GraphsTuple with the following fields:
            - nodes.positions
            - nodes.species
            - senders
            - receivers

        Returns:
            node_embeddings: an array of shape [num_nodes, output_irreps]
        """
        relative_positions = (
            graphs.nodes.positions[graphs.receivers]
            - graphs.nodes.positions[graphs.senders]
        )
        relative_positions = e3nn.IrrepsArray("1o", relative_positions)
        species = graphs.nodes.species
        num_nodes = species.shape[0]

        node_embeddings: e3nn.IrrepsArray = mace_jax.modules.MACE(
            output_irreps=self.output_irreps,
            r_max=self.r_max,
            num_interactions=self.num_interactions,
            hidden_irreps=self.hidden_irreps,
            readout_mlp_irreps=self.readout_mlp_irreps,
            avg_num_neighbors=self.avg_num_neighbors,
            num_species=self.num_species,
            radial_basis=lambda x, x_max: e3nn.bessel(x, self.num_basis_fns, x_max),
            radial_envelope=e3nn.soft_envelope,
            max_ell=self.max_ell,
            skip_connection_first_layer=False,
            soft_normalization=self.soft_normalization,
        )(relative_positions, species, graphs.senders, graphs.receivers)

        assert node_embeddings.shape == (
            num_nodes,
            self.num_interactions,
            node_embeddings.irreps.dim,
        )
        node_embeddings = node_embeddings.axis_to_mul(axis=-2)
        return node_embeddings
