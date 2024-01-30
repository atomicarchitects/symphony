from typing import Callable, Optional
import jax
import jax.numpy as jnp
import haiku as hk
import e3nn_jax as e3nn

from symphony import datatypes
from symphony.models import ptable


def shifted_softplus(x: jnp.ndarray) -> jnp.ndarray:
    """A softplus function shifted so that shifted_softplus(0) = 0."""
    return jax.nn.softplus(x) - jnp.log(2.0)


def cosine_cutoff(input: jnp.ndarray, cutoff: jnp.ndarray):
    """Behler-style cosine cutoff, adapted from SchNetPack."""
    # Compute values of cutoff function
    input_cut = 0.5 * (jnp.cos(input * jnp.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).astype(jnp.float32)
    return input_cut


class E3SchNetInteractionBlock(hk.Module):
    r"""E(3)-equivariant SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        num_filters: int,
        max_ell: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
    ):
        """
        Args:
            num_filters: number of filters used in continuous-filter convolution.
            max_ell: maximal ell for spherical harmonics.
            activation: if None, no activation function is used.
        """
        super(E3SchNetInteractionBlock, self).__init__()
        self.num_filters = num_filters
        self.max_ell = max_ell
        self.activation = activation

    def __call__(
        self,
        x: e3nn.IrrepsArray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        f_ij: jnp.ndarray,
        rcut_ij: jnp.ndarray,
        Yr_ij: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        """Compute interaction output. Notation matches SchNetPack implementation in PyTorch.
        Args:
            x: input IrrepsArray indicating node features
            idx_i: index of center atom i
            idx_j: index of neighbors j
            f_ij: d_ij passed through the embedding function
            rcut_ij: d_ij passed through the cutoff function
            r_ij: relative position of neighbor j to atom i
            Yr_ij: spherical harmonics of r_ij
        Returns:
            atom features after interaction
        """
        input_irreps = x.irreps

        # Embed the inputs.
        x = e3nn.haiku.Linear(
            irreps_out=self.num_filters * e3nn.Irreps.spherical_harmonics(self.max_ell)
        )(x)

        # Select senders.
        x_j = x[idx_j]
        x_j = x_j.mul_to_axis(self.num_filters, axis=-2)
        x_j = e3nn.tensor_product(x_j, Yr_ij)
        x_j = x_j.axis_to_mul(axis=-2)

        # Compute filter.
        W_ij = hk.Sequential(
            [
                hk.Linear(self.num_filters),
                lambda x: self.activation(x),
                hk.Linear(x_j.irreps.num_irreps),
            ]
        )(f_ij)
        W_ij = W_ij * rcut_ij[:, None]
        W_ij = e3nn.IrrepsArray(f"{x_j.irreps.num_irreps}x0e", W_ij)

        # Compute continuous-filter convolution.
        x_ij = x_j * W_ij
        x = e3nn.scatter_sum(x_ij, dst=idx_i, output_size=x.shape[0])

        # Apply final linear and activation layers.
        x = e3nn.haiku.Linear(
            irreps_out=input_irreps,
        )(x)
        x = e3nn.scalar_activation(
            x,
            acts=[self.activation if ir.l == 0 else None for _, ir in input_irreps],
        )
        x = e3nn.haiku.Linear(irreps_out=input_irreps)(x)
        return x


class E3SchNet(hk.Module):
    """A Haiku implementation of E3SchNet."""

    def __init__(
        self,
        init_embedding_dim: int,
        num_interactions: int,
        num_filters: int,
        num_radial_basis_functions: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        cutoff: float,
        max_ell: int,
        num_species: int,
        name: Optional[str] = None,
        simple_embedding: bool = True
    ):
        """
        Args:
            init_embedding_dim: the size of the initial embedding for atoms
            num_interactions: number of interaction blocks
            num_filters: number of filters used in continuous-filter convolution
            num_radial_basis_functions: number of radial basis functions
            activation: activation function
            cutoff: cutoff radius
            max_ell: maximal ell for spherical harmonics
            num_species: number of species
        """
        super().__init__(name=name)
        self.init_embedding_dim = init_embedding_dim
        self.num_interactions = num_interactions
        self.activation = activation
        self.num_filters = num_filters
        self.num_radial_basis_functions = num_radial_basis_functions
        self.radial_basis = lambda x: e3nn.soft_one_hot_linspace(
            x,
            start=0.0,
            end=cutoff,
            number=self.num_radial_basis_functions,
            basis="gaussian",
            cutoff=True,
        )
        self.cutoff_fn = lambda x: cosine_cutoff(x, cutoff=cutoff)
        self.max_ell = max_ell
        self.num_species = num_species
        self.simple_embedding = simple_embedding

    def __call__(self, fragments: datatypes.Fragments) -> jnp.ndarray:
        # 'species' are actually atomic numbers mapped to [0, self.num_species).
        # But we keep the same name for consistency with SchNetPack.
        species = fragments.nodes.species
        r_ij = (
            fragments.nodes.positions[fragments.receivers]
            - fragments.nodes.positions[fragments.senders]
        )
        idx_i = fragments.receivers
        idx_j = fragments.senders

        # Irreps for the quantities we need to compute.]
        spherical_harmonics_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)
        latent_irreps = e3nn.Irreps(
            (self.init_embedding_dim, (ir.l, ir.p))
            for _, ir in spherical_harmonics_irreps
        )

        # Compute atom embeddings.
        # Initially, the atom embeddings are just scalars.
        if self.simple_embedding:
            x = hk.Embed(self.num_species, self.init_embedding_dim)(species)
        else:
            x_species = hk.Embed(self.num_species, self.init_embedding_dim)(species)
            x_group = hk.Embed(18, self.init_embedding_dim)(jax.vmap(lambda s: ptable.groups[s])(species))
            x_row = hk.Embed(7, self.init_embedding_dim)(jax.vmap(lambda s: ptable.rows[s])(species))
            x_block = hk.Embed(4, self.init_embedding_dim)(jax.vmap(lambda s: ptable.blocks[s])(species))
            x = x_species + x_group + x_row + x_block  # TODO: what's the best way to combine these things?
        x = e3nn.IrrepsArray(f"{x.shape[-1]}x0e", x)
        x = e3nn.haiku.Linear(irreps_out=latent_irreps, force_irreps_out=True)(x)

        # Compute radial basis functions to cut off interactions
        d_ij = jnp.linalg.norm(r_ij, axis=-1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)
        r_ij = r_ij * rcut_ij[:, None]

        # Compute the spherical harmonics of relative positions.
        # r_ij: (n_edges, 3)
        # Yr_ij: (n_edges, (max_ell + 1) ** 2)
        # Reshape Yr_ij to (num_edges, 1, (max_ell + 1) ** 2).
        Yr_ij = e3nn.spherical_harmonics(
            spherical_harmonics_irreps, r_ij, normalize=True, normalization="component"
        )
        Yr_ij = Yr_ij.reshape((Yr_ij.shape[0], 1, Yr_ij.shape[1]))

        # Compute interaction block to update atomic embeddings
        for _ in range(self.num_interactions):
            v = E3SchNetInteractionBlock(
                self.num_filters, self.max_ell, self.activation
            )(x, idx_i, idx_j, f_ij, rcut_ij, Yr_ij)
            x = x + v
        # In SchNetPack, the output is only the scalar features.
        # Here, we return the entire IrrepsArray.
        return x
