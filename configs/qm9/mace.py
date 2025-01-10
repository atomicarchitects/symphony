"""Defines the default hyperparameters and training configuration for the MACE model."""

import ml_collections

from configs.qm9 import default


def get_embedder_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the MACE model."""
    config = ml_collections.ConfigDict()

    # MACE hyperparameters.
    config.model = "MACE"
    config.num_hidden_channels = 16
    config.num_channels = 128
    config.r_max = 5
    config.num_interactions = 1
    config.avg_num_neighbors = 15.0
    config.max_ell = 3
    config.num_basis_fns = 8
    config.activation = "softplus"
    config.soft_normalization = 1e2
    config.use_pseudoscalars_and_pseudovectors = True

    return config


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the MACE model."""
    config = default.get_config()

    config.focus_and_target_species_predictor.embedder_config = get_embedder_config()
    config.target_position_predictor.embedder_config = get_embedder_config()

    return config
