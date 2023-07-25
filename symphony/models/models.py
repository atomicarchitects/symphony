"""Definition of the generative models."""

from typing import Callable, Optional, Tuple

import chex
import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import ml_collections

from symphony import datatypes
from symphony.models import nequip, marionette, e3schnet, mace, attention, allegro

ATOMIC_NUMBERS = [1, 6, 7, 8, 9]
NUM_ELEMENTS = len(ATOMIC_NUMBERS)


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
    log_coeffs: e3nn.IrrepsArray, res_beta: int, res_alpha: int, num_radii: int
) -> e3nn.SphericalSignal:
    """Converts coefficients of the logits to a SphericalSignal representing the logits."""
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
            global_embeddings = attention.MultiHeadAttention(
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
        min_radius: float,
        max_radius: float,
        num_radii: int,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.position_coeffs_lmax = position_coeffs_lmax
        self.res_beta = res_beta
        self.res_alpha = res_alpha
        self.num_channels = num_channels
        self.num_species = num_species
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_radii = num_radii

    def create_radii(self) -> jnp.ndarray:
        """Creates the binned radii for the target positions."""
        return jnp.linspace(self.min_radius, self.max_radius, self.num_radii)

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
        position_coeffs = e3nn.haiku.Linear(
            self.num_radii * self.num_channels * irreps, force_irreps_out=True
        )(target_species_embeddings * focus_node_embeddings)
        position_coeffs = position_coeffs.mul_to_axis(factor=self.num_channels)
        position_coeffs = position_coeffs.mul_to_axis(factor=self.num_radii)
        assert position_coeffs.shape == (
            num_graphs,
            self.num_channels,
            self.num_radii,
            irreps.dim,
        )

        # Compute the position signal projected to a spherical grid for each radius.
        position_logits = jax.vmap(
            lambda coeffs: log_coeffs_to_logits(
                coeffs, self.res_beta, self.res_alpha, self.num_radii
            )
        )(position_coeffs)
        assert position_logits.shape == (
            num_graphs,
            self.num_radii,
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
        auxiliary_node_embedder: hk.Module,
        focus_and_target_species_predictor: FocusAndTargetSpeciesPredictor,
        target_position_predictor: TargetPositionPredictor,
        global_embedder: Optional[GlobalEmbedder] = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.node_embedder = node_embedder
        self.auxiliary_node_embedder = auxiliary_node_embedder
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

        # Get the species and stop probabilities.
        focus_and_target_species_probs, stop_probs = segment_softmax_2D_with_stop(
            focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
        )

        # Get the embeddings of the focus nodes.
        # These are the first nodes in each graph during training.
        auxiliary_node_embeddings = self.auxiliary_node_embedder(graphs)
        focus_node_indices = get_first_node_indices(graphs)
        true_focus_node_embeddings = auxiliary_node_embeddings[focus_node_indices]

        # Get the position coefficients.
        position_coeffs, position_logits = self.target_position_predictor(
            true_focus_node_embeddings, graphs.globals.target_species
        )

        # Get the position probabilities.
        # For numerical stability, we subtract out the maximum value over all spheres before exponentiating.
        # Our loss accounts for unnormalized probabilities.
        position_max = jnp.max(
            position_logits.grid_values, axis=(-3, -2, -1), keepdims=True
        )
        position_max = jax.lax.stop_gradient(position_max)
        position_probs = position_logits.apply(
            lambda pos: jnp.exp(pos - position_max)
        )  # [num_graphs, num_radii, res_beta, res_alpha]

        # The radii bins used for the position prediction, repeated for each graph.
        radii = self.target_position_predictor.create_radii()
        radii_bins = jnp.tile(radii, (num_graphs, 1))

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
            self.target_position_predictor.num_radii,
            position_coeffs.shape[-1],
        ), position_coeffs.shape
        assert position_logits.shape == (
            num_graphs,
            self.target_position_predictor.num_radii,
            self.target_position_predictor.res_beta,
            self.target_position_predictor.res_alpha,
        )

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings=node_embeddings,
                auxiliary_node_embeddings=auxiliary_node_embeddings,
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
                radii_bins=radii_bins,
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

        # We stop a graph, if we sample a stop.
        rng, stop_rng = jax.random.split(rng)
        stop = jax.random.bernoulli(stop_rng, stop_probs)

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
        auxiliary_node_embeddings = self.auxiliary_node_embedder(graphs)
        focus_node_embeddings = auxiliary_node_embeddings[focus_indices]

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

        # Sample the radius.
        radii = self.target_position_predictor.create_radii()
        radii_bins = jnp.tile(radii, (num_graphs, 1))
        radii_probs = position_probs.integrate().array.squeeze(
            axis=-1
        )  # [num_graphs, num_radii]
        num_radii = radii.shape[0]
        rng, radius_rng = jax.random.split(rng)
        radius_rngs = jax.random.split(radius_rng, num_graphs)
        radius_indices = jax.vmap(
            lambda key, p: jax.random.choice(key, num_radii, p=p)
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
            lambda r, b, a: radii[r] * angular_probs.grid_vectors[b, a]
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
            num_radii,
            irreps.dim,
        )
        assert position_logits.shape == (
            num_graphs,
            self.target_position_predictor.num_radii,
            res_beta,
            res_alpha,
        )
        assert position_vectors.shape == (num_graphs, 3)

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings=node_embeddings,
                auxiliary_node_embeddings=auxiliary_node_embeddings,
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
                radii_bins=radii_bins,
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
            return e3schnet.shifted_softplus
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

            def node_embedder_fn():
                output_irreps = config.num_channels * e3nn.s2_irreps(config.max_ell)
                return mace.MACE(
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

            def node_embedder_fn():
                irreps = e3nn.s2_irreps(config.max_ell)
                if config.use_pseudoscalars_and_pseudovectors:
                    irreps += e3nn.Irreps("0o + 1e")
                output_irreps = config.num_channels * irreps
                return nequip.NequIP(
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
                    skip_connection=config.skip_connection,
                )

        elif config.model == "MarioNette":

            def node_embedder_fn():
                return marionette.MarioNette(
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

            def node_embedder_fn():
                return e3schnet.E3SchNet(
                    n_atom_basis=config.num_channels,
                    n_interactions=config.num_interactions,
                    n_filters=config.num_channels,
                    n_rbf=config.num_basis_fns,
                    activation=get_activation(config.activation),
                    cutoff=config.cutoff,
                    max_ell=config.max_ell,
                    num_species=num_species,
                )

        elif config.model == "Allegro":

            def node_embedder_fn():
                irreps = e3nn.s2_irreps(config.max_ell)
                if config.use_pseudoscalars_and_pseudovectors:
                    irreps += e3nn.Irreps("0o + 1e")
                output_irreps = config.num_channels * irreps

                return allegro.Allegro(
                    num_species=num_species,
                    r_max=config.r_max,
                    avg_num_neighbors=config.avg_num_neighbors,
                    max_ell=config.max_ell,
                    output_irreps=output_irreps,
                    num_interactions=config.num_interactions,
                    mlp_activation=get_activation(config.mlp_activation),
                    mlp_n_hidden=config.num_channels,
                    mlp_n_layers=config.mlp_n_layers,
                    n_radial_basis=config.num_basis_fns,
                )

        else:
            raise ValueError(f"Unsupported model: {config.model}.")

        # Create the node embedders.
        node_embedder = node_embedder_fn()
        auxiliary_node_embedder = node_embedder_fn()

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
            min_radius=config.target_position_predictor.min_radius,
            max_radius=config.target_position_predictor.max_radius,
            num_radii=config.target_position_predictor.num_radii,
        )
        predictor = Predictor(
            node_embedder=node_embedder,
            auxiliary_node_embedder=auxiliary_node_embedder,
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
