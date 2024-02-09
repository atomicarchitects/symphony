"""Defines the default hyperparameters and training configuration for the E3SchNet model."""

import ml_collections

from configs.linker import default


def get_embedder_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the E3SchNet model."""
    config = ml_collections.ConfigDict()
    # E3SchNet hyperparameters.
    config.model = "E3SchNet"
    config.cutoff = 5.0
    config.num_interactions = 3
    config.num_filters = 16
    config.num_radial_basis_functions = 8
    config.num_channels = 64
    config.max_ell = 2
    config.activation = "shifted_softplus"
    config.simple_embedding = True

    return config


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()

    config.focus_and_target_species_predictor.embedder_config = get_embedder_config()
    config.target_position_predictor.embedder_config = get_embedder_config()

    # NequIP hyperparameters.
    return config
