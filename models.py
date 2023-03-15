"""Definition of the generative models."""

from typing import Callable, Optional, Sequence, Tuple, Union

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import mace_jax.modules
import nequip_jax
import flax.linen as nn
import chex
import functools

import datatypes

RADII = jnp.arange(0.75, 2.03, 0.02)
NUM_ELEMENTS = 5


def get_first_node_indices(graphs: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the indices of the focus nodes in each graph."""
    return jnp.concatenate((jnp.asarray([0]), jnp.cumsum(graphs.n_node)[:-1]))


@functools.partial(jax.jit, static_argnames="num_segments")
def segment_sample(
    probabilities: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: int,
    rng: chex.PRNGKey,
):
    """Sample indices from a categorical distribution across each segment.
    Args:
        probabilities: A 1D array of probabilities.
        segment_ids: A 1D array of segment ids.
        num_segments: The number of segments.
        rng: A PRNG key.
    Returns:
        A 1D array of sampled indices.
    """

    def sample_for_segment(rng, i):
        return jax.random.choice(
            rng, node_indices, p=jnp.where(i == segment_ids, probabilities, 0.0)
        )

    node_indices = jnp.arange(len(segment_ids))
    rngs = jax.random.split(rng, num_segments)
    return jax.vmap(sample_for_segment)(rngs, jnp.arange(num_segments))


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

    def __call__(self, graphs: jraph.GraphsTuple) -> jnp.ndarray:
        # 'species' are actually atomic numbers mapped to [0, NUM_ELEMENTS).
        # But we keep the same name for consistency with SchNetPack.
        atomic_numbers = graphs.nodes.species
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
        x = hk.Embed(NUM_ELEMENTS, self.n_atom_basis)(atomic_numbers)
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
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.output_irreps = e3nn.Irreps(output_irreps)
        self.r_max = r_max
        self.num_interactions = num_interactions
        self.hidden_irreps = hidden_irreps
        self.readout_mlp_irreps = readout_mlp_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.num_species = num_species
        self.max_ell = max_ell
        self.num_basis_fns = num_basis_fns

    def __call__(self, graphs: datatypes.Fragment) -> e3nn.IrrepsArray:
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
        relative_positions = e3nn.IrrepsArray(e3nn.Irreps('1o'), relative_positions)
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
        )(relative_positions, species, graphs.senders, graphs.receivers)

        assert node_embeddings.shape == (
            num_nodes,
            self.num_interactions,
            self.output_irreps.dim,
        )
        node_embeddings = node_embeddings.axis_to_mul(axis=1)
        return node_embeddings


class FocusPredictor(hk.Module):
    """Predicts focus logits for each node."""

    def __init__(
        self,
        latent_size: int,
        num_layers: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.activation = activation

    def __call__(self, node_embeddings: e3nn.IrrepsArray) -> jnp.ndarray:
        node_scalars = node_embeddings.filter(keep="0e")
        focus_logits = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[self.latent_size] * (self.num_layers - 1) + [1],
            act=self.activation,
        )(node_scalars)
        focus_logits = focus_logits.array.squeeze(axis=-1)
        return focus_logits


class TargetSpeciesPredictor(hk.Module):
    """Predicts the target species conditioned on the focus node embeddings."""

    def __init__(
        self,
        latent_size: int,
        num_layers: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.activation = activation

    def __call__(self, focus_node_embeddings: e3nn.IrrepsArray) -> jnp.ndarray:
        focus_node_scalars = focus_node_embeddings.filter(keep="0e")
        species_logits = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[self.latent_size] * (self.num_layers - 1) + [NUM_ELEMENTS],
            act=self.activation,
        )(focus_node_scalars).array
        return species_logits


class TargetPositionPredictor(hk.Module):
    """Predicts the position coefficients for the target species."""

    position_coeffs_lmax: int

    def __init__(self, position_coeffs_lmax: int, name: Optional[str] = None):
        super().__init__(name)
        self.position_coeffs_lmax = position_coeffs_lmax

    def __call__(
        self, focus_node_embeddings: e3nn.IrrepsArray, target_species: jnp.ndarray
    ) -> e3nn.IrrepsArray:
        num_graphs = focus_node_embeddings.shape[0]

        assert focus_node_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.dim,
        )

        irreps = e3nn.s2_irreps(self.position_coeffs_lmax, p_val=1, p_arg=-1)
        target_species_embeddings = hk.Embed(
            NUM_ELEMENTS, focus_node_embeddings.irreps.num_irreps
        )(target_species)

        assert target_species_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.num_irreps,
        )

        # TODO (ameyad, mariogeiger, songk): See if we can make this more expressive.
        position_coeffs = e3nn.haiku.Linear(len(RADII) * irreps)(
            target_species_embeddings * focus_node_embeddings
        )
        position_coeffs = position_coeffs.mul_to_axis(factor=len(RADII))
        return position_coeffs


class Predictor(hk.Module):
    node_embedder: hk.Module
    focus_predictor: hk.Module
    target_species_predictor: hk.Module
    target_position_predictor: hk.Module
    run_in_evaluation_mode: bool

    def __init__(
        self,
        node_embedder: hk.Module,
        focus_predictor: hk.Module,
        target_species_predictor: hk.Module,
        target_position_predictor: hk.Module,
        run_in_evaluation_mode: bool,
        name: str = None,
    ):
        super().__init__(name=name)
        self.node_embedder = node_embedder
        self.focus_predictor = focus_predictor
        self.target_species_predictor = target_species_predictor
        self.target_position_predictor = target_position_predictor
        self.run_in_evaluation_mode = run_in_evaluation_mode

    def get_training_predictions(
        self, graphs: datatypes.Fragment
    ) -> datatypes.Predictions:
        """Returns the predictions on these graphs during training, when we have access to the true focus and target species."""
        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]

        # Get the node embeddings.
        node_embeddings = self.node_embedder(graphs)

        # Get the focus logits.
        focus_logits = self.focus_predictor(node_embeddings)

        # Get the embeddings of the focus nodes.
        # These are the first nodes in each graph during training.
        focus_node_indices = get_first_node_indices(graphs)
        true_focus_node_embeddings = node_embeddings[focus_node_indices]

        # Get the species logits.
        target_species_logits = self.target_species_predictor(
            true_focus_node_embeddings
        )

        # Get the position coefficients.
        position_coeffs = self.target_position_predictor(
            true_focus_node_embeddings, graphs.globals.target_species
        )

        # Check the shapes.
        assert focus_logits.shape == (num_nodes,)
        assert target_species_logits.shape == (num_graphs, NUM_ELEMENTS)
        assert position_coeffs.shape[:2] == (num_graphs, len(RADII))

        return datatypes.Predictions(
            focus_logits=focus_logits,
            target_species_logits=target_species_logits,
            position_coeffs=position_coeffs,
        )

    def get_evaluation_predictions(
        self, graphs: datatypes.Fragment
    ) -> datatypes.EvaluationPredictions:
        """Returns the predictions on a single padded graph during evaluation, when we do not have access to the true focus and target species."""
        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]

        # Get the node embeddings.
        node_embeddings = self.node_embedder(graphs)

        # Compute corresponding focus probabilities.
        focus_logits = self.focus_predictor(node_embeddings)
        focus_probs = jraph.partition_softmax(focus_logits, graphs.n_node, num_nodes)

        # Get the PRNG key.
        rng = hk.next_rng_key()

        # Sample the focus node.
        rng, focus_rng = jax.random.split(rng)
        segment_ids = jnp.repeat(
            jnp.arange(num_graphs), graphs.n_node, axis=0, total_repeat_length=num_nodes
        )
        focus_indices = segment_sample(focus_probs, segment_ids, num_graphs, focus_rng)

        # Get the embeddings of the focus node.
        focus_node_embeddings = node_embeddings[focus_indices]

        # Get the species logits.
        target_species_logits = self.target_species_predictor(focus_node_embeddings)
        target_species_probs = jax.nn.softmax(target_species_logits)

        # Sample the target species.
        rng, species_rng = jax.random.split(rng)
        species_rngs = jax.random.split(species_rng, num_graphs)
        target_species = jax.vmap(
            lambda key, p: jax.random.choice(key, NUM_ELEMENTS, p=p)
        )(species_rngs, target_species_probs)

        # Get the position coefficients.
        position_coeffs = self.target_position_predictor(
            focus_node_embeddings, target_species
        )

        return datatypes.EvaluationPredictions(
            focus_logits=focus_logits,
            focus_indices=focus_indices,
            target_species_logits=target_species_logits,
            position_coeffs=position_coeffs,
            target_species=target_species,
        )

    def __call__(self, graphs: datatypes.Fragment) -> datatypes.Predictions:
        if self.run_in_evaluation_mode:
            return self.get_evaluation_predictions(graphs)
        return self.get_training_predictions(graphs)

class NequIP(nn.Module):
    """Wrapper class for NequIP."""

    latent_size: int
    avg_num_neighbors: float
    sh_lmax: int
    target_irreps: e3nn.Irreps
    even_activation: Callable[[jnp.ndarray], jnp.ndarray]
    odd_activation: Callable[[jnp.ndarray], jnp.ndarray]
    mlp_activation: Callable[[jnp.ndarray], jnp.ndarray]
    mlp_n_hidden: int
    mlp_n_layers: int
    n_radial_basis: int

    @nn.compact
    def __call__(
        self,
        graphs: datatypes.Fragment,
    ):
        species_embedder = nn.Embed(NUM_ELEMENTS, self.latent_size)

        # Predict the properties.
        vectors = (
            graphs.nodes.positions[graphs.receivers]
            - graphs.nodes.positions[graphs.senders]
        )
        vectors = e3nn.IrrepsArray(e3nn.Irreps('1o'), vectors)
        species = graphs.nodes.species
        n_nodes = graphs.nodes.positions.shape[0]
        node_feats = nn.Embed(n_nodes, self.latent_size)(species)
        node_feats = e3nn.IrrepsArray(f"{node_feats.shape[1]}x0e", node_feats)
        
        node_embeddings = nequip_jax.NEQUIPLayer(
            avg_num_neighbors=self.avg_num_neighbors,
            num_species=NUM_ELEMENTS,
            sh_lmax=self.sh_lmax,
            target_irreps=self.target_irreps,
            even_activation=self.even_activation,
            odd_activation=self.odd_activation,
            mlp_activation=self.mlp_activation,
            mlp_n_hidden=self.mlp_n_hidden,
            mlp_n_layers=self.mlp_n_layers,
            n_radial_basis=self.n_radial_basis,
        )(
            vectors,
            node_feats,
            species,
            graphs.senders,
            graphs.receivers
        )
        true_focus_node_embeddings = node_embeddings[get_first_node_indices(graphs)]
        target_species_embeddings = species_embedder(graphs.globals.target_species)

        focus_logits = nn.Dense(1)(node_embeddings).squeeze(axis=-1)
        species_logits = nn.Dense(NUM_ELEMENTS)(true_focus_node_embeddings)

        irreps = e3nn.s2_irreps(self.position_coeffs_lmax, p_val=1, p_arg=-1)
        input_for_position_coeffs = jnp.concatenate(
            (true_focus_node_embeddings, target_species_embeddings), axis=-1
        )
        position_coeffs = nn.Dense(len(RADII) * irreps.dim)(
            input_for_position_coeffs
        )
        position_coeffs = jnp.reshape(position_coeffs, (-1, len(RADII), irreps.dim))
        position_coeffs = e3nn.IrrepsArray(irreps=irreps, array=position_coeffs)

        return datatypes.Predictions(
            focus_logits=focus_logits,
            species_logits=species_logits,
            position_coeffs=position_coeffs,
        )
