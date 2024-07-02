
from typing import Callable
import ml_collections

import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import haiku as hk

from symphony import datatypes
from symphony.data import datasets
from symphony.models.angular_predictors.linear_angular_predictor import (
    LinearAngularPredictor,
)
from symphony.models.radius_predictors.discretized_predictor import (
    DiscretizedRadialPredictor,
)
from symphony.models.radius_predictors.rational_quadratic_spline import (
    RationalQuadraticSplineRadialPredictor,
)
from symphony.models.position_predictor import TargetPositionPredictor as DiscretizedTargetPositionPredictor
from symphony.models.continuous_position_predictor import TargetPositionPredictor
from symphony.models.predictor import Predictor
from symphony.models.focus_predictor import FocusAndTargetSpeciesPredictor
from symphony.models.embedders import nequip, marionette, e3schnet, mace, allegro


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


def create_node_embedder(
    config: ml_collections.ConfigDict,
    num_species: int,
) -> hk.Module:
    """Creates a node embedder as specified by the config."""

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
            periodic_table_embedding=config.periodic_table_embedding,
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
        )

    raise ValueError(f"Unsupported model: {config.model}.")


def create_predictor(config: ml_collections.ConfigDict) -> Predictor:
    """Creates the predictor model as specified by the config."""

    num_species = datasets.utils.get_dataset(config).num_species()
    focus_and_target_species_predictor = FocusAndTargetSpeciesPredictor(
        node_embedder_fn=lambda: create_node_embedder(
            config.focus_and_target_species_predictor.embedder_config,
            num_species,
        ),
        latent_size=config.focus_and_target_species_predictor.latent_size,
        num_layers=config.focus_and_target_species_predictor.num_layers,
        activation=get_activation(
            config.focus_and_target_species_predictor.activation
        ),
        num_species=num_species,
    )
    angular_predictor_config = config.target_position_predictor.angular_predictor
    radial_predictor_config = config.target_position_predictor.radial_predictor
    angular_predictor_fn = lambda: LinearAngularPredictor(
        max_ell=config.target_position_predictor.embedder_config.max_ell,
        num_channels=angular_predictor_config.num_channels,
        radial_mlp_num_layers=angular_predictor_config.radial_mlp_num_layers,
        radial_mlp_latent_size=angular_predictor_config.radial_mlp_latent_size,
        max_radius=radial_predictor_config.max_radius,
        res_beta=angular_predictor_config.res_beta,
        res_alpha=angular_predictor_config.res_alpha,
        quadrature=angular_predictor_config.quadrature,
        sampling_inverse_temperature_factor=angular_predictor_config.sampling_inverse_temperature_factor,
        sampling_num_steps=angular_predictor_config.sampling_num_steps,
        sampling_init_step_size=angular_predictor_config.sampling_init_step_size,
    )
    if config.target_position_predictor.radial_predictor_type == "rational_quadratic_spline":
        radial_predictor_fn = lambda: RationalQuadraticSplineRadialPredictor(
            num_bins=radial_predictor_config.num_bins,
            min_radius=radial_predictor_config.min_radius,
            max_radius=radial_predictor_config.max_radius,
            num_layers=radial_predictor_config.num_layers,
            num_param_mlp_layers=radial_predictor_config.num_param_mlp_layers,
            boundary_error=radial_predictor_config.boundary_error,
        )
    elif config.target_position_predictor.radial_predictor_type == "discretized":
        radial_predictor_fn = lambda: DiscretizedRadialPredictor(
            num_bins=radial_predictor_config.num_bins,
            range_min=radial_predictor_config.min_radius,
            range_max=radial_predictor_config.max_radius,
            num_layers=radial_predictor_config.num_layers,
            latent_size=radial_predictor_config.latent_size,
        )
    else:
        raise ValueError(
            f"Unsupported radial predictor type: {config.target_position_predictor.radial_predictor_type}."
        )
    target_position_predictor = TargetPositionPredictor(
        node_embedder_fn=lambda: create_node_embedder(
            config.target_position_predictor.embedder_config,
            num_species,
        ),
        angular_predictor_fn=angular_predictor_fn,
        radial_predictor_fn=radial_predictor_fn,
        num_species=num_species,
    )
    predictor = Predictor(
        focus_and_target_species_predictor=focus_and_target_species_predictor,
        target_position_predictor=target_position_predictor,
    )
    return predictor


def create_model(
    config: ml_collections.ConfigDict, run_in_evaluation_mode: bool
) -> hk.Transformed:
    """Create a model with init() and apply() defined, as specified by the config."""

    def model_fn(
        graphs: datatypes.Fragments,
        focus_and_atom_type_inverse_temperature: float = 1.0,
        position_inverse_temperature: float = 1.0,
    ) -> datatypes.Predictions:
        """Defines the entire network."""
        predictor = create_predictor(config)
        if run_in_evaluation_mode:
            return predictor.get_evaluation_predictions(
                graphs,
                focus_and_atom_type_inverse_temperature,
                position_inverse_temperature,
            )
        else:
            return predictor.get_training_predictions(graphs)

    return hk.transform(model_fn)
