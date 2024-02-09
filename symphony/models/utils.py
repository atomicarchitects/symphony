"""Definition of the generative models."""

from typing import Callable, Optional, Tuple, Union

import chex
import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import ml_collections

from symphony import datatypes
from symphony.models.predictor import Predictor
from symphony.models.embedders.global_embedder import GlobalEmbedder
from symphony.models.focus_predictor import FocusAndTargetSpeciesPredictor
from symphony.models.position_predictor import (
    TargetPositionPredictor,
    FactorizedTargetPositionPredictor,
)
from symphony.models.position_updater import PositionUpdater
from symphony.models.embedders import nequip, marionette, e3schnet, mace, allegro

# ATOMIC_NUMBERS = list(range(1, 84))  # QCD
# ATOMIC_NUMBERS = list(range(1, 81))  # TMQM
# ATOMIC_NUMBERS = [1, 6, 7, 8, 9]  # QM9
ATOMIC_NUMBERS = [6, 7, 8, 9, 16, 17, 35, 53]  # linker


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
    ), f"{log_coeffs.shape}"

    log_dist = e3nn.to_s2grid(
        log_coeffs, res_beta, res_alpha, quadrature="gausslegendre", p_val=1, p_arg=-1
    )
    assert log_dist.shape == (num_channels, num_radii, res_beta, res_alpha)

    # Combine over all channels.
    log_dist.grid_values = jax.scipy.special.logsumexp(log_dist.grid_values, axis=0)
    assert log_dist.shape == (num_radii, res_beta, res_alpha)

    # Subtract the max to avoid numerical issues.
    max_logit = jnp.max(log_dist.grid_values)
    max_logit = jax.lax.stop_gradient(max_logit)
    log_dist.grid_values -= max_logit

    return log_dist


def position_logits_to_position_distribution(
    position_logits: e3nn.SphericalSignal,
) -> e3nn.SphericalSignal:
    """Converts logits to a SphericalSignal representing the position distribution."""

    assert len(position_logits.shape) == 3  # [num_radii, res_beta, res_alpha]
    max_logit = jnp.max(position_logits.grid_values)
    max_logit = jax.lax.stop_gradient(max_logit)

    position_probs = position_logits.apply(lambda logit: jnp.exp(logit - max_logit))

    normalizing_factor = position_probs.integrate().array.sum()
    position_probs.grid_values /= normalizing_factor
    return position_probs


def safe_log(x: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """Computes the log of x, replacing 0 with a small value for numerical stability."""
    return jnp.log(jnp.where(x == 0, eps, x))


def position_distribution_to_radial_distribution(
    position_probs: e3nn.SphericalSignal,
) -> jnp.ndarray:
    """Computes the marginal radial distribution from a logits of a distribution over all positions."""
    assert len(position_probs.shape) == 3  # [num_radii, res_beta, res_alpha]
    return position_probs.integrate().array.squeeze(axis=-1)  # [..., num_radii]


def position_distribution_to_angular_distribution(
    position_probs: e3nn.SphericalSignal,
) -> jnp.ndarray:
    """Returns the marginal radial distribution for a logits of a distribution over all positions."""
    assert len(position_probs.shape) == 3  # [num_radii, res_beta, res_alpha]
    position_probs.grid_values = position_probs.grid_values.sum(axis=0)
    return position_probs


def compute_grid_of_joint_distribution(
    radial_weights: jnp.ndarray,
    log_angular_coeffs: e3nn.IrrepsArray,
    res_beta: int,
    res_alpha: int,
    quadrature: str,
) -> e3nn.SphericalSignal:
    """Combines radial weights and angular coefficients to get a distribution on the spheres.
    Should theoretically only be used for distributions over a single position."""
    # Convert coefficients to a distribution on the sphere.
    log_angular_dist = e3nn.to_s2grid(
        log_angular_coeffs,
        res_beta,
        res_alpha,
        quadrature=quadrature,
        p_val=1,
        p_arg=-1,
    )

    # Subtract the maximum value for numerical stability.
    log_angular_dist_max = jnp.max(
        log_angular_dist.grid_values, axis=(-2, -1), keepdims=True
    )
    log_angular_dist_max = jax.lax.stop_gradient(log_angular_dist_max)
    log_angular_dist = log_angular_dist.apply(lambda x: x - log_angular_dist_max)

    # Convert to a probability distribution, by taking the exponential and normalizing.
    angular_dist = log_angular_dist.apply(jnp.exp)
    angular_dist = angular_dist / angular_dist.integrate()

    # Check that shapes are correct.
    num_radii = radial_weights.shape[0]
    assert angular_dist.shape == (
        res_beta,
        res_alpha,
    ), angular_dist.shape

    # Mix in the radius weights to get a distribution over all spheres.
    dist = e3nn.SphericalSignal(
        grid_values=jnp.einsum(
            "r, ba -> rba", radial_weights, angular_dist.grid_values
        ),
        quadrature=angular_dist.quadrature,
    )
    assert dist.shape == (num_radii, res_beta, res_alpha)
    return dist  # [num_radii, res_beta, res_alpha]


def compute_coefficients_of_logits_of_joint_distribution(
    radial_logits: jnp.ndarray,
    log_angular_coeffs: e3nn.IrrepsArray,
) -> e3nn.IrrepsArray:
    """Combines radial weights and angular coefficients to get a distribution on the spheres."""
    radial_logits = e3nn.IrrepsArray("0e", radial_logits[:, None])
    log_dist_coeffs = jax.vmap(
        lambda log_radial_weight: e3nn.concatenate(
            [log_radial_weight, log_angular_coeffs]
        )
    )(radial_logits)
    log_dist_coeffs = e3nn.sum(log_dist_coeffs.regroup(), axis=-1)

    num_radii = radial_logits.shape[0]
    assert log_dist_coeffs.shape == (num_radii, log_dist_coeffs.irreps.dim)

    return log_dist_coeffs


def get_activation(activation: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Get the activation function."""
    if activation == "shifted_softplus":
        return e3schnet.shifted_softplus
    return getattr(jax.nn, activation)


def _irreps_from_lmax(
    lmax: int, num_channels: int, use_pseudoscalars_and_pseudovectors: bool
) -> e3nn.Irreps:
    """Convenience function to create irreps from lmax."""
    irreps = e3nn.s2_irreps(lmax)
    if use_pseudoscalars_and_pseudovectors:
        irreps += e3nn.Irreps("0o + 1e")
    return (num_channels * irreps).regroup()


def get_num_species_for_dataset(dataset: str) -> int:
    """Returns the number of species for a given dataset."""
    if dataset in ["qm9", "tmqm", "linker"]:
        return len(ATOMIC_NUMBERS)
    if dataset in ["tetris", "platonic_solids"]:
        return 1
    raise ValueError(f"Unsupported dataset: {dataset}.")


def create_node_embedder(
    config: ml_collections.ConfigDict,
    num_species: int,
    name_prefix: Optional[str] = None,
) -> hk.Module:
    if name_prefix is None:
        raise ValueError("name_prefix must be specified.")

    if config.model == "MACE":
        output_irreps = _irreps_from_lmax(
            config.max_ell,
            config.num_channels,
            config.use_pseudoscalars_and_pseudovectors,
        )
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
            name=f"node_embedder_{name_prefix}_mace",
        )

    if config.model == "NequIP":
        output_irreps = _irreps_from_lmax(
            config.max_ell,
            config.num_channels,
            config.use_pseudoscalars_and_pseudovectors,
        )
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
            name=f"node_embedder_{name_prefix}_nequip",
        )

    if config.model == "MarioNette":
        output_irreps = _irreps_from_lmax(
            config.max_ell,
            config.num_channels,
            config.use_pseudoscalars_and_pseudovectors,
        )
        return marionette.MarioNette(
            num_species=num_species,
            r_max=config.r_max,
            avg_num_neighbors=config.avg_num_neighbors,
            init_embedding_dims=config.num_channels,
            output_irreps=output_irreps,
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
            name=f"node_embedder_{name_prefix}_marionette",
        )

    if config.model == "E3SchNet":
        return e3schnet.E3SchNet(
            init_embedding_dim=config.num_channels,
            num_interactions=config.num_interactions,
            num_filters=config.num_filters,
            num_radial_basis_functions=config.num_radial_basis_functions,
            activation=get_activation(config.activation),
            cutoff=config.cutoff,
            max_ell=config.max_ell,
            num_species=num_species,
            name=f"node_embedder_{name_prefix}_e3schnet",
        )

    if config.model == "Allegro":
        output_irreps = _irreps_from_lmax(
            config.max_ell,
            config.num_channels,
            config.use_pseudoscalars_and_pseudovectors,
        )
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
            name=f"node_embedder_{name_prefix}_allegro",
        )

    raise ValueError(f"Unsupported model: {config.model}.")


def create_position_updater(
    config: ml_collections.ConfigDict,
) -> hk.Transformed:
    """Create a position updater as specified by the config."""
    dataset = config.get("dataset", "qm9")
    num_species = get_num_species_for_dataset(dataset)

    def model_fn(graphs: datatypes.Fragments):
        return PositionUpdater(
            node_embedder=create_node_embedder(
                config.position_updater.embedder_config,
                num_species,
                name_prefix="position_updater",
            )
        )(graphs)

    return hk.transform(model_fn)


def create_model(
    config: ml_collections.ConfigDict, run_in_evaluation_mode: bool
) -> hk.Transformed:
    """Create a model as specified by the config."""

    if config.get("position_updater"):
        return create_position_updater(config)

    def model_fn(
        graphs: datatypes.Fragments,
        focus_and_atom_type_inverse_temperature: float = 1.0,
        position_inverse_temperature: float = 1.0,
    ) -> datatypes.Predictions:
        """Defines the entire network."""

        dataset = config.get("dataset", "qm9")
        num_species = get_num_species_for_dataset(dataset)

        if config.focus_and_target_species_predictor.get("compute_global_embedding"):
            global_embedder = GlobalEmbedder(
                num_channels=config.focus_and_target_species_predictor.global_embedder.num_channels,
                pooling=config.focus_and_target_species_predictor.global_embedder.pooling,
                num_attention_heads=config.focus_and_target_species_predictor.global_embedder.num_attention_heads,
            )
        else:
            global_embedder = None

        focus_and_target_species_predictor = FocusAndTargetSpeciesPredictor(
            node_embedder=create_node_embedder(
                config.focus_and_target_species_predictor.embedder_config,
                num_species,
                name_prefix="focus_and_target_species_predictor",
            ),
            global_embedder=global_embedder,
            latent_size=config.focus_and_target_species_predictor.latent_size,
            num_layers=config.focus_and_target_species_predictor.num_layers,
            activation=get_activation(
                config.focus_and_target_species_predictor.activation
            ),
            num_species=num_species,
        )
        if config.target_position_predictor.get("factorized"):
            target_position_predictor = FactorizedTargetPositionPredictor(
                node_embedder=create_node_embedder(
                    config.target_position_predictor.embedder_config,
                    num_species,
                    name_prefix="target_position_predictor",
                ),
                position_coeffs_lmax=config.target_position_predictor.embedder_config.max_ell,
                res_beta=config.target_position_predictor.res_beta,
                res_alpha=config.target_position_predictor.res_alpha,
                num_channels=config.target_position_predictor.num_channels,
                num_species=num_species,
                min_radius=config.target_position_predictor.min_radius,
                max_radius=config.target_position_predictor.max_radius,
                num_radii=config.target_position_predictor.num_radii,
                radial_mlp_latent_size=config.target_position_predictor.radial_mlp_latent_size,
                radial_mlp_num_layers=config.target_position_predictor.radial_mlp_num_layers,
                radial_mlp_activation=get_activation(
                    config.target_position_predictor.radial_mlp_activation
                ),
                apply_gate=config.target_position_predictor.get("apply_gate"),
            )
        else:
            target_position_predictor = TargetPositionPredictor(
                node_embedder=create_node_embedder(
                    config.target_position_predictor.embedder_config,
                    num_species,
                    name_prefix="target_position_predictor",
                ),
                position_coeffs_lmax=config.target_position_predictor.embedder_config.max_ell,
                res_beta=config.target_position_predictor.res_beta,
                res_alpha=config.target_position_predictor.res_alpha,
                num_channels=config.target_position_predictor.num_channels,
                num_species=num_species,
                min_radius=config.target_position_predictor.min_radius,
                max_radius=config.target_position_predictor.max_radius,
                num_radii=config.target_position_predictor.num_radii,
                apply_gate=config.target_position_predictor.get("apply_gate"),
            )

        predictor = Predictor(
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
