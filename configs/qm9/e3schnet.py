"""Defines the default hyperparameters and training configuration for the E3SchNet model."""

import ml_collections

from configs.qm9 import default


def get_embedder_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the E3SchNet model."""
    config = ml_collections.ConfigDict()

    config.model = "E3SchNet"
    config.cutoff = 5.0
    config.num_interactions = 3
    config.num_filters = 16
    config.num_radial_basis_functions = 8
    config.num_channels = 32
    config.max_ell = 2
    config.activation = "shifted_softplus"
    config.simple_embedding = False

    return config


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()

    config.focus_and_target_species_predictor.embedder_config = get_embedder_config()
    config.target_position_predictor.embedder_config = get_embedder_config()

    return config
