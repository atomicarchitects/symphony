"""Definition of the generative models."""

from typing import Callable, Optional, Tuple

import chex
import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import jraph
import mace_jax.modules
import ml_collections
import nequip_jax

import datatypes
import marionette

RADII = jnp.arange(0.5, 1.5, 0.05)
ATOMIC_NUMBERS = [1, 6, 7, 8, 9]
NUM_ELEMENTS = len(ATOMIC_NUMBERS)


def debug_print(fmt, *args, **kwargs):
    return
    jax.debug.print(fmt, *args, **kwargs)


def get_atomic_numbers(species: jnp.ndarray) -> jnp.ndarray:
    """Returns the atomic numbers for the species."""
    return jnp.asarray(ATOMIC_NUMBERS)[species]


def get_first_node_indices(graphs: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the indices of the focus nodes in each graph."""
    return jnp.concatenate((jnp.asarray([0]), jnp.cumsum(graphs.n_node)[:-1]))


def segment_softmax_2D_with_stop(
    species_logits: jnp.ndarray,
    stop_logits: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the species probabilities and stop probabilities with segment softmax over 2D arrays of species logits."""
    # Subtract the max to avoid numerical issues.
    logits_max = jraph.segment_max(
        species_logits, segment_ids, num_segments=num_segments
    ).max(axis=-1)
    logits_max = jnp.maximum(logits_max, stop_logits)
    logits_max = jax.lax.stop_gradient(logits_max)
    species_logits -= logits_max[segment_ids, None]
    stop_logits -= logits_max

    # Normalize exp() by all nodes, all atom types, and the stop for each graph.
    exp_species_logits = jnp.exp(species_logits)
    exp_species_logits_summed = jnp.sum(exp_species_logits, axis=-1)
    normalizing_factors = jraph.segment_sum(
        exp_species_logits_summed, segment_ids, num_segments=num_segments
    )
    exp_stop_logits = jnp.exp(stop_logits)

    normalizing_factors += exp_stop_logits
    species_probs = exp_species_logits / normalizing_factors[segment_ids, None]
    stop_probs = exp_stop_logits / normalizing_factors

    return species_probs, stop_probs


def get_segment_ids(
    n_node: jnp.ndarray,
    num_nodes: int,
) -> jnp.ndarray:
    """Returns the segment ids for each node in the graphs."""
    num_graphs = n_node.shape[0]

    return jnp.repeat(
        jnp.arange(num_graphs), n_node, axis=0, total_repeat_length=num_nodes
    )


def segment_sample_2D(
    species_probabilities: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    rng: chex.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample indices from a categorical distribution across each segment.
    Args:
        species_probabilities: A 2D array of probabilities.
        segment_ids: A 1D array of segment ids.
        num_segments: The number of segments.
        rng: A PRNG key.
    Returns:
        A 1D array of sampled indices, one for each segment.
    """
    num_nodes, num_species = species_probabilities.shape

    # Normalize the probabilities to sum up for 1 over all nodes in each graph.
    species_probabilities_summed = jraph.segment_sum(
        species_probabilities.sum(axis=-1), segment_ids, num_segments
    )
    species_probabilities = (
        species_probabilities / species_probabilities_summed[segment_ids, None]
    )

    def sample_for_segment(rng: chex.PRNGKey, segment_id: int) -> Tuple[float, float]:
        """Samples a node and species index for a single segment."""
        node_rng, logit_rng, rng = jax.random.split(rng, num=3)
        node_index = jax.random.choice(
            node_rng,
            jnp.arange(num_nodes),
            p=jnp.where(
                segment_id == segment_ids, species_probabilities.sum(axis=-1), 0.0
            ),
        )
        normalized_probs_for_index = species_probabilities[node_index] / jnp.sum(
            species_probabilities[node_index]
        )
        species_index = jax.random.choice(
            logit_rng, jnp.arange(num_species), p=normalized_probs_for_index
        )
        return node_index, species_index

    rngs = jax.random.split(rng, num_segments)
    node_indices, species_indices = jax.vmap(sample_for_segment)(
        rngs, jnp.arange(num_segments)
    )
    assert node_indices.shape == (num_segments,)
    assert species_indices.shape == (num_segments,)
    return node_indices, species_indices


def log_coeffs_to_logits(
    log_coeffs: e3nn.IrrepsArray, res_beta: int, res_alpha: int,
) -> e3nn.SphericalSignal:
    """Converts coefficients of the logits to a SphericalSignal representing the logits."""
    num_radii = len(RADII)
    num_channels = log_coeffs.shape[0]
    assert log_coeffs.shape == (
        num_channels,
        num_radii,
        log_coeffs.irreps.dim,
    ), log_coeffs.shape

    log_dist = e3nn.to_s2grid(
        log_coeffs, res_beta, res_alpha, quadrature="gausslegendre", p_val=1, p_arg=-1
    )
    assert log_dist.shape == (num_channels, num_radii, res_beta, res_alpha)

    # Softmax over all channels.
    log_dist.grid_values = jax.scipy.special.logsumexp(log_dist.grid_values, axis=0)
    return log_dist


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
        n_atom_basis: int,
        n_filters: int,
        max_ell: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(E3SchNetInteractionBlock, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters
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
            irreps_out=self.n_filters * e3nn.Irreps.spherical_harmonics(self.max_ell)
        )(x)

        # Select senders.
        x_j = x[idx_j]
        x_j = x_j.mul_to_axis(self.n_filters, axis=-2)
        x_j = e3nn.tensor_product(x_j, Yr_ij)
        x_j = x_j.axis_to_mul(axis=-2)

        # Compute filter.
        W_ij = hk.Sequential(
            [
                hk.Linear(self.n_filters),
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
        n_atom_basis: int,
        n_interactions: int,
        n_filters: int,
        n_rbf: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        cutoff: float,
        max_ell: int,
        num_species: int,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.activation = activation
        self.n_filters = n_filters
        self.n_rbf = n_rbf
        self.radial_basis = lambda x: e3nn.soft_one_hot_linspace(
            x,
            start=0.0,
            end=cutoff,
            number=self.n_rbf,
            basis="gaussian",
            cutoff=True,
        )
        self.cutoff_fn = lambda x: cosine_cutoff(x, cutoff=cutoff)
        self.max_ell = max_ell
        self.num_species = num_species

    def __call__(self, graphs: jraph.GraphsTuple) -> jnp.ndarray:
        # 'species' are actually atomic numbers mapped to [0, self.num_species).
        # But we keep the same name for consistency with SchNetPack.
        species = graphs.nodes.species
        r_ij = (
            graphs.nodes.positions[graphs.receivers]
            - graphs.nodes.positions[graphs.senders]
        )
        idx_i = graphs.receivers
        idx_j = graphs.senders

        # Irreps for the quantities we need to compute.
        spherical_harmonics_irreps = e3nn.Irreps.spherical_harmonics(self.max_ell)
        latent_irreps = e3nn.Irreps(
            (self.n_atom_basis, (ir.l, ir.p)) for _, ir in spherical_harmonics_irreps
        )

        # Compute atom embeddings.
        # Initially, the atom embeddings are just scalars.
        x = hk.Embed(self.num_species, self.n_atom_basis)(species)
        x = e3nn.IrrepsArray(f"{x.shape[-1]}x0e", x)
        x = e3nn.haiku.Linear(irreps_out=latent_irreps)(x)

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
        for _ in range(self.n_interactions):
            v = E3SchNetInteractionBlock(
                self.n_atom_basis, self.n_filters, self.max_ell, self.activation
            )(x, idx_i, idx_j, f_ij, rcut_ij, Yr_ij)
            x = x + v

        # In SchNetPack, the output is only the scalar features.
        # Here, we return the entire IrrepsArray.
        return x


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
            skip_connection_first_layer=True,
            soft_normalization=self.soft_normalization,
        )(relative_positions, species, graphs.senders, graphs.receivers)

        assert node_embeddings.shape == (
            num_nodes,
            self.num_interactions,
            self.output_irreps.dim,
        )
        node_embeddings = node_embeddings.axis_to_mul(axis=-2)
        return node_embeddings


class NequIP(hk.Module):
    """Wrapper class for NequIP."""

    def __init__(
        self,
        num_species: int,
        r_max: float,
        avg_num_neighbors: float,
        max_ell: int,
        init_embedding_dims: int,
        output_irreps: str,
        num_interactions: int,
        even_activation: Callable[[jnp.ndarray], jnp.ndarray],
        odd_activation: Callable[[jnp.ndarray], jnp.ndarray],
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
        self.init_embedding_dims = init_embedding_dims
        self.output_irreps = output_irreps
        self.num_interactions = num_interactions
        self.even_activation = even_activation
        self.odd_activation = odd_activation
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

        species = graphs.nodes.species
        node_feats = hk.Embed(self.num_species, self.init_embedding_dims)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)

        for _ in range(self.num_interactions):
            node_feats = nequip_jax.NEQUIPESCNLayerHaiku(
                avg_num_neighbors=self.avg_num_neighbors,
                num_species=self.num_species,
                output_irreps=self.output_irreps,
                even_activation=self.even_activation,
                odd_activation=self.odd_activation,
                mlp_activation=self.mlp_activation,
                mlp_n_hidden=self.mlp_n_hidden,
                mlp_n_layers=self.mlp_n_layers,
                n_radial_basis=self.n_radial_basis,
            )(relative_positions, node_feats, species, graphs.senders, graphs.receivers)

        alpha = 0.5 ** jnp.array(node_feats.irreps.ls)
        node_feats = node_feats * alpha
        return node_feats


class MarioNette(hk.Module):
    """Wrapper class for MarioNette."""

    def __init__(
        self,
        num_species: int,
        r_max: float,
        avg_num_neighbors: float,
        init_embedding_dims: int,
        output_irreps: str,
        soft_normalization: float,
        num_interactions: int,
        even_activation: Callable[[jnp.ndarray], jnp.ndarray],
        odd_activation: Callable[[jnp.ndarray], jnp.ndarray],
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray],
        mlp_n_hidden: int,
        mlp_n_layers: int,
        n_radial_basis: int,
        use_bessel: bool,
        alpha: float,
        alphal: float,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_species = num_species
        self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.init_embedding_dims = init_embedding_dims
        self.output_irreps = output_irreps
        self.soft_normalization = soft_normalization
        self.num_interactions = num_interactions
        self.even_activation = even_activation
        self.odd_activation = odd_activation
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.n_radial_basis = n_radial_basis
        self.use_bessel = use_bessel
        self.alpha = alpha
        self.alphal = alphal

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

        for _ in range(self.num_interactions):
            node_feats = marionette.MarioNetteLayerHaiku(
                avg_num_neighbors=self.avg_num_neighbors,
                num_species=self.num_species,
                output_irreps=self.output_irreps,
                interaction_irreps=self.output_irreps,
                soft_normalization=self.soft_normalization,
                even_activation=self.even_activation,
                odd_activation=self.odd_activation,
                mlp_activation=self.mlp_activation,
                mlp_n_hidden=self.mlp_n_hidden,
                mlp_n_layers=self.mlp_n_layers,
                n_radial_basis=self.n_radial_basis,
                use_bessel=self.use_bessel,
            )(relative_positions, node_feats, species, graphs.senders, graphs.receivers)

        alpha = self.alpha * (self.alphal ** jnp.array(node_feats.irreps.ls))
        node_feats = node_feats * alpha
        return node_feats


class MultiHeadAttention(hk.Module):
    """Multi-headed attention (MHA) module.

    This module is intended for attending over sequences of IrrepsArrays.

    Rough sketch:
    - Compute keys (K), queries (Q), and values (V) as projections of inputs.
    - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
    - Output is another projection of WV^T.

    For more detail, see the original Transformer paper:
      "Attention is all you need" https://arxiv.org/abs/1706.03762.

    Glossary of shapes:
    - T: Sequence length.
    - D: Vector (embedding) size.
    - H: Number of attention heads.
    """

    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        name: Optional[str] = None,
    ):
        """Initialises the module.

        Args:
          num_heads: Number of independent attention heads (H).
          num_channels: The number of channels in each head to determine the embedding size (D).
          name: Optional name for this module.
        """
        super().__init__(name=name)
        self.num_heads = num_heads
        self.num_channels = num_channels

    def __call__(
        self,
        query: e3nn.IrrepsArray,
        key: e3nn.IrrepsArray,
        value: e3nn.IrrepsArray,
        mask: Optional[jnp.ndarray] = None,
    ) -> e3nn.IrrepsArray:
        """Computes (optionally masked) MHA with queries, keys & values.

        This module broadcasts over zero or more 'batch-like' leading dimensions.

        Args:
          query: Embeddings sequence used to compute queries; shape [..., T', D_q].
          key: Embeddings sequence used to compute keys; shape [..., T, D_k].
          value: Embeddings sequence used to compute values; shape [..., T, D_v].
          mask: Optional mask applied to attention weights; shape [..., H=1, T', T].

        Returns:
          A new sequence of embeddings, consisting of a projection of the
            attention-weighted value projections; shape [..., T', D'].
        """
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        irreps = query.irreps
        *leading_dims, sequence_length, _ = query.shape
        projection = self._linear_projection

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        query_heads = projection(
            query, self.num_heads, self.num_channels, "query"
        )  # [T', H, Q]
        key_heads = projection(
            key, self.num_heads, self.num_channels, "key"
        )  # [T, H, K]
        value_heads = projection(
            value, self.num_heads, self.num_channels, "value"
        )  # [T, H, V]

        # Compute attention weights.
        attn_logits = jnp.einsum(
            "...thd,...Thd->...htT", query_heads.array, key_heads.array
        )
        attn_logits = attn_logits / jnp.sqrt(query_heads.shape[-1])
        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads.array)
        attn = e3nn.IrrepsArray(self.num_channels * irreps, attn)  # [T', H, V]
        attn = attn.axis_to_mul(axis=-2)  # [T', H * V]

        # Apply another projection to get the final embeddings.
        return e3nn.haiku.Linear(self.num_channels * irreps)(attn)  # [T', D']

    @hk.transparent
    def _linear_projection(
        self,
        x: e3nn.IrrepsArray,
        num_heads: int,
        num_channels: int,
        name: Optional[str] = None,
    ) -> jnp.ndarray:
        y = e3nn.haiku.Linear(num_channels * num_heads * x.irreps, name=name)(x)
        y = y.mul_to_axis(num_heads, axis=-2)
        return y


class GlobalEmbedder(hk.Module):
    """Computes a global embedding for each node in the graph."""

    def __init__(
        self,
        num_channels: int,
        pooling: str,
        num_attention_heads: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.num_channels = num_channels
        self.pooling = pooling
        if self.pooling == "attention":
            assert num_attention_heads is not None
            self.num_attention_heads = num_attention_heads
        else:
            assert num_attention_heads is None

    def __call__(self, graphs: datatypes.Fragments) -> jnp.ndarray:
        node_embeddings: e3nn.IrrepsArray = graphs.nodes
        num_nodes = node_embeddings.shape[0]
        num_graphs = graphs.n_node.shape[0]
        irreps = node_embeddings.irreps
        segment_ids = get_segment_ids(graphs.n_node, num_nodes)

        if self.pooling == "mean":
            global_embeddings = jraph.segment_mean(
                node_embeddings.array, segment_ids, num_segments=num_graphs
            )
            global_embeddings = e3nn.IrrepsArray(irreps, global_embeddings)
            global_embeddings = e3nn.haiku.Linear(self.num_channels * irreps)(
                global_embeddings
            )

        elif self.pooling == "sum":
            global_embeddings = jraph.segment_sum(
                node_embeddings.array, segment_ids, num_segments=num_graphs
            )
            global_embeddings = e3nn.IrrepsArray(irreps, global_embeddings)
            global_embeddings = e3nn.haiku.Linear(self.num_channels * irreps)(
                global_embeddings
            )

        elif self.pooling == "attention":
            # Only attend to nodes within the same graph.
            attention_mask = jnp.where(
                segment_ids[:, None] == segment_ids[None, :], 1.0, 0.0
            )
            attention_mask = jnp.expand_dims(attention_mask, axis=0)
            global_embeddings = MultiHeadAttention(
                self.num_attention_heads, self.num_channels
            )(node_embeddings, node_embeddings, node_embeddings, attention_mask)

        assert global_embeddings.shape == (num_nodes, self.num_channels * irreps.dim)
        return global_embeddings


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


class TargetPositionPredictor(hk.Module):
    """Predicts the position coefficients for the target species."""

    def __init__(
        self,
        position_coeffs_lmax: int,
        res_beta: int,
        res_alpha: int,
        num_channels: int,
        num_species: int,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.position_coeffs_lmax = position_coeffs_lmax
        self.res_beta = res_beta
        self.res_alpha = res_alpha
        self.num_channels = num_channels
        self.num_species = num_species

    def __call__(
        self, focus_node_embeddings: e3nn.IrrepsArray, target_species: jnp.ndarray
    ) -> Tuple[e3nn.IrrepsArray, e3nn.SphericalSignal]:
        num_graphs = focus_node_embeddings.shape[0]

        assert focus_node_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.dim,
        )

        target_species_embeddings = hk.Embed(
            self.num_species, embed_dim=focus_node_embeddings.irreps.num_irreps
        )(target_species)

        assert target_species_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.num_irreps,
        )

        # TODO: See if we can make this more expressive.
        irreps = e3nn.s2_irreps(self.position_coeffs_lmax, p_val=1, p_arg=-1)
        num_radii = len(RADII)
        position_coeffs = e3nn.haiku.Linear(num_radii * self.num_channels * irreps)(
            target_species_embeddings * focus_node_embeddings
        )
        position_coeffs = position_coeffs.mul_to_axis(factor=self.num_channels)
        position_coeffs = position_coeffs.mul_to_axis(factor=num_radii)
        assert position_coeffs.shape == (
            num_graphs,
            self.num_channels,
            num_radii,
            irreps.dim,
        ), position_coeffs.shape

        # Old logic:
        # position_logits = e3nn.to_s2grid(position_coeffs)

        # Compute the position signal projected to a spherical grid for each radius.
        position_logits = jax.vmap(
            lambda coeffs: log_coeffs_to_logits(
                coeffs, self.res_beta, self.res_alpha
            )
        )(position_coeffs)
        assert position_logits.shape == (
            num_graphs,
            num_radii,
            self.res_beta,
            self.res_alpha,
        ), position_logits.shape

        position_logits.grid_values -= position_logits.grid_values.max(
            axis=(-3, -2, -1), keepdims=True
        )

        return position_coeffs, position_logits


class Predictor(hk.Module):
    """A convenient wrapper for an entire prediction model."""

    def __init__(
        self,
        node_embedder: hk.Module,
        focus_and_target_species_predictor: FocusAndTargetSpeciesPredictor,
        target_position_predictor: TargetPositionPredictor,
        global_embedder: Optional[GlobalEmbedder] = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.node_embedder = node_embedder
        self.global_embedder = global_embedder
        self.focus_and_target_species_predictor = focus_and_target_species_predictor
        self.target_position_predictor = target_position_predictor

    def get_training_predictions(
        self, graphs: datatypes.Fragments
    ) -> datatypes.Predictions:
        """Returns the predictions on these graphs during training, when we have access to the true focus and target species."""
        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]
        num_species = self.focus_and_target_species_predictor.num_species
        segment_ids = get_segment_ids(graphs.n_node, num_nodes)

        # Get the node embeddings.
        node_embeddings = self.node_embedder(graphs)

        # Concatenate global embeddings to node embeddings.
        if self.global_embedder is not None:
            graphs_with_node_embeddings = graphs._replace(nodes=node_embeddings)
            global_embeddings = self.global_embedder(graphs_with_node_embeddings)
            node_embeddings = e3nn.concatenate(
                [node_embeddings, global_embeddings], axis=-1
            )


        # Get the species and stop logits.
        focus_and_target_species_logits = self.focus_and_target_species_predictor(
            node_embeddings
        )
        stop_logits = jnp.zeros((num_graphs,))

        focus_and_target_species_probs, stop_probs = segment_softmax_2D_with_stop(
            focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
        )

        # Get the embeddings of the focus nodes.
        # These are the first nodes in each graph during training.
        focus_node_indices = get_first_node_indices(graphs)
        true_focus_node_embeddings = node_embeddings[focus_node_indices]

        # Get the position coefficients.
        position_coeffs, position_logits = self.target_position_predictor(
            true_focus_node_embeddings, graphs.globals.target_species
        )

        # Integrate the position signal over each sphere to get the normalizing factors for the radii.
        # For numerical stability, we subtract out the maximum value over all spheres before exponentiating.
        position_max = jnp.max(
            position_logits.grid_values, axis=(-3, -2, -1), keepdims=True
        )
        position_max = jax.lax.stop_gradient(position_max)
        position_probs = position_logits.apply(
            lambda pos: jnp.exp(pos - position_max)
        )  # [num_graphs, num_radii, res_beta, res_alpha]
        # position_probs = position_probs / position_probs.integrate()

        # Check the shapes.
        assert focus_and_target_species_logits.shape == (
            num_nodes,
            num_species,
        ), focus_and_target_species_logits.shape
        assert focus_and_target_species_probs.shape == (
            num_nodes,
            num_species,
        ), focus_and_target_species_probs.shape
        assert position_coeffs.shape == (
            num_graphs,
            self.target_position_predictor.num_channels,
            len(RADII),
            position_coeffs.shape[-1],
        ), position_coeffs.shape
        assert position_logits.shape == (
            num_graphs,
            len(RADII),
            self.target_position_predictor.res_beta,
            self.target_position_predictor.res_alpha,
        )

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings=node_embeddings,
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop_logits=stop_logits,
                stop_probs=stop_probs,
                stop=None,
                focus_indices=focus_node_indices,
                target_species=None,
                position_coeffs=position_coeffs,
                position_logits=position_logits,
                position_probs=position_probs,
                position_vectors=None,
            ),
            senders=graphs.senders,
            receivers=graphs.receivers,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )

    def get_evaluation_predictions(
        self,
        graphs: datatypes.Fragments,
        focus_and_atom_type_inverse_temperature: float,
        position_inverse_temperature: float,
    ) -> datatypes.Predictions:
        """Returns the predictions on a single padded graph during evaluation, when we do not have access to the true focus and target species."""
        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]
        num_species = self.focus_and_target_species_predictor.num_species
        segment_ids = get_segment_ids(graphs.n_node, num_nodes)

        # Get the PRNG key for sampling.
        rng = hk.next_rng_key()

        # Get the node embeddings.
        node_embeddings = self.node_embedder(graphs)

        # Concatenate global embeddings to node embeddings.
        if self.global_embedder is not None:
            graphs_with_node_embeddings = graphs._replace(nodes=node_embeddings)
            global_embeddings = self.global_embedder(graphs_with_node_embeddings)
            node_embeddings = e3nn.concatenate(
                [node_embeddings, global_embeddings], axis=-1
            )

        # Get the species and stop logits.
        focus_and_target_species_logits = self.focus_and_target_species_predictor(
            node_embeddings
        )
        stop_logits = jnp.zeros((num_graphs,))

        # Scale the logits by the inverse temperature.
        focus_and_target_species_logits *= focus_and_atom_type_inverse_temperature
        stop_logits *= focus_and_atom_type_inverse_temperature

        # Get the softmaxed probabilities.
        focus_and_target_species_probs, stop_probs = segment_softmax_2D_with_stop(
            focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
        )

        # We stop a graph if the stop probability is greater than 0.5.
        stop = stop_probs > 0.5

        # Renormalize the focus and target species probabilities, if we have not stopped.
        focus_and_target_species_probs = focus_and_target_species_probs / (
            (1 - stop_probs)[segment_ids, None]
        )

        # Sample the focus node and target species.
        rng, focus_rng = jax.random.split(rng)
        focus_indices, target_species = segment_sample_2D(
            focus_and_target_species_probs, segment_ids, num_graphs, focus_rng
        )

        # Get the embeddings of the focus node.
        focus_node_embeddings = node_embeddings[focus_indices]

        # Get the position coefficients.
        position_coeffs, position_logits = self.target_position_predictor(
            focus_node_embeddings, target_species
        )

        # Scale by inverse temperature.
        position_coeffs = position_inverse_temperature * position_coeffs
        position_logits = position_inverse_temperature * position_logits

        # Integrate the position signal over each sphere to get the normalizing factors for the radii.
        # For numerical stability, we subtract out the maximum value over all spheres before exponentiating.
        max_logit = jnp.max(
            position_logits.grid_values, axis=(-3, -2, -1), keepdims=True
        )
        max_logit = jax.lax.stop_gradient(max_logit)
        position_probs = position_logits.apply(
            lambda logit: jnp.exp(logit - max_logit)
        )  # [num_graphs, num_radii, res_beta, res_alpha]

        radii_probs = position_probs.integrate().array.squeeze(
            axis=-1
        )  # [num_graphs, num_radii]

        # Sample the radius.
        rng, radius_rng = jax.random.split(rng)
        radius_rngs = jax.random.split(radius_rng, num_graphs)
        radius_indices = jax.vmap(
            lambda key, p: jax.random.choice(key, len(RADII), p=p)
        )(
            radius_rngs, radii_probs
        )  # [num_graphs]

        # Get the angular probabilities.
        angular_probs = jax.vmap(
            lambda p, r_index: p[r_index] / p[r_index].integrate()
        )(
            position_probs, radius_indices
        )  # [num_graphs, res_beta, res_alpha]

        # Sample angles.
        rng, angular_rng = jax.random.split(rng)
        angular_rngs = jax.random.split(angular_rng, num_graphs)
        beta_indices, alpha_indices = jax.vmap(lambda key, p: p.sample(key))(
            angular_rngs, angular_probs
        )

        # Combine the radius and angles to get the position vectors.
        position_vectors = jax.vmap(
            lambda r, b, a: RADII[r] * angular_probs.grid_vectors[b, a]
        )(radius_indices, beta_indices, alpha_indices)

        # Check the shapes.
        irreps = e3nn.s2_irreps(self.target_position_predictor.position_coeffs_lmax)
        res_beta, res_alpha = (
            self.target_position_predictor.res_beta,
            self.target_position_predictor.res_alpha,
        )

        assert stop.shape == (num_graphs,)
        assert focus_indices.shape == (num_graphs,)
        assert focus_and_target_species_logits.shape == (num_nodes, num_species)
        assert focus_and_target_species_probs.shape == (num_nodes, num_species)
        assert position_coeffs.shape == (
            num_graphs,
            self.target_position_predictor.num_channels,
            len(RADII),
            irreps.dim,
        )
        assert position_logits.shape == (num_graphs, len(RADII), res_beta, res_alpha)
        assert position_vectors.shape == (num_graphs, 3)

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings=node_embeddings,
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop_logits=stop_logits,
                stop_probs=stop_probs,
                stop=stop,
                focus_indices=focus_indices,
                target_species=target_species,
                position_coeffs=position_coeffs,
                position_logits=position_logits,
                position_probs=position_probs,
                position_vectors=position_vectors,
            ),
            senders=graphs.senders,
            receivers=graphs.receivers,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )


def create_model(
    config: ml_collections.ConfigDict, run_in_evaluation_mode: bool
) -> hk.Transformed:
    """Create a model as specified by the config."""

    def get_activation(activation: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Get the activation function."""

        if activation == "shifted_softplus":
            return shifted_softplus
        return getattr(jax.nn, activation)

    def model_fn(
        graphs: datatypes.Fragments,
        focus_and_atom_type_inverse_temperature: float = 1.0,
        position_inverse_temperature: float = 1.0,
    ) -> datatypes.Predictions:
        """Defines the entire network."""

        dataset = config.get("dataset", "qm9")
        if dataset == "qm9":
            num_species = NUM_ELEMENTS
        if dataset in ["tetris", "platonic_solids"]:
            num_species = 1

        if config.model == "MACE":
            output_irreps = config.num_channels * e3nn.s2_irreps(config.max_ell)
            node_embedder = MACE(
                output_irreps=output_irreps,
                hidden_irreps=output_irreps,
                readout_mlp_irreps=output_irreps,
                r_max=config.r_max,
                num_interactions=config.num_interactions,
                avg_num_neighbors=config.avg_num_neighbors,
                num_species=num_species,
                max_ell=config.max_ell,
                num_basis_fns=config.num_basis_fns,
                soft_normalization=config.get("soft_normalization"),
            )
        elif config.model == "NequIP":
            output_irreps = config.num_channels * e3nn.s2_irreps(config.max_ell)
            node_embedder = NequIP(
                num_species=num_species,
                r_max=config.r_max,
                avg_num_neighbors=config.avg_num_neighbors,
                max_ell=config.max_ell,
                init_embedding_dims=config.num_channels,
                output_irreps=output_irreps,
                num_interactions=config.num_interactions,
                even_activation=get_activation(config.even_activation),
                odd_activation=get_activation(config.odd_activation),
                mlp_activation=get_activation(config.mlp_activation),
                mlp_n_hidden=config.num_channels,
                mlp_n_layers=config.mlp_n_layers,
                n_radial_basis=config.num_basis_fns,
            )
        elif config.model == "MarioNette":
            node_embedder = MarioNette(
                num_species=num_species,
                r_max=config.r_max,
                avg_num_neighbors=config.avg_num_neighbors,
                init_embedding_dims=config.num_channels,
                output_irreps=config.num_channels * e3nn.s2_irreps(config.max_ell),
                soft_normalization=config.soft_normalization,
                num_interactions=config.num_interactions,
                even_activation=get_activation(config.even_activation),
                odd_activation=get_activation(config.odd_activation),
                mlp_activation=get_activation(config.activation),
                mlp_n_hidden=config.num_channels,
                mlp_n_layers=config.mlp_n_layers,
                n_radial_basis=config.num_basis_fns,
                use_bessel=config.use_bessel,
                alpha=config.alpha,
                alphal=config.alphal,
            )
        elif config.model == "E3SchNet":
            node_embedder = E3SchNet(
                n_atom_basis=config.num_channels,
                n_interactions=config.num_interactions,
                n_filters=config.num_channels,
                n_rbf=config.num_basis_fns,
                activation=get_activation(config.activation),
                cutoff=config.cutoff,
                max_ell=config.max_ell,
                num_species=num_species,
            )
        else:
            raise ValueError(f"Unsupported model: {config.model}.")

        if config.compute_global_embedding:
            global_embedder = GlobalEmbedder(
                num_channels=config.global_embedder.num_channels,
                pooling=config.global_embedder.pooling,
                num_attention_heads=config.global_embedder.num_attention_heads,
            )
        else:
            global_embedder = None

        focus_and_target_species_predictor = FocusAndTargetSpeciesPredictor(
            latent_size=config.focus_and_target_species_predictor.latent_size,
            num_layers=config.focus_and_target_species_predictor.num_layers,
            activation=get_activation(config.activation),
            num_species=num_species,
        )
        target_position_predictor = TargetPositionPredictor(
            position_coeffs_lmax=config.max_ell,
            res_beta=config.target_position_predictor.res_beta,
            res_alpha=config.target_position_predictor.res_alpha,
            num_channels=config.target_position_predictor.num_channels,
            num_species=num_species,
        )
        predictor = Predictor(
            node_embedder=node_embedder,
            global_embedder=global_embedder,
            focus_and_target_species_predictor=focus_and_target_species_predictor,
            target_position_predictor=target_position_predictor,
        )

        if run_in_evaluation_mode:
            return predictor.get_evaluation_predictions(
                graphs,
                focus_and_atom_type_inverse_temperature,
                position_inverse_temperature,
            )
        else:
            return predictor.get_training_predictions(graphs)

    return hk.transform(model_fn)
