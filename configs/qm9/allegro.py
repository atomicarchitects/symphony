"""Defines the default hyperparameters and training configuration for the Allegro model."""

import ml_collections

from configs.qm9 import default


def get_embedder_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the Allegro model."""
    config = ml_collections.ConfigDict()

    config.model = "Allegro"
    config.num_channels = 64
    config.r_max = 5
    config.avg_num_neighbors = 300.0  # Allegro is not properly normalized.
    config.num_interactions = 2
    config.max_ell = 4
    config.mlp_activation = "swish"
    config.activation = "softplus"
    config.mlp_n_layers = 2
    config.num_basis_fns = 8
    config.use_pseudoscalars_and_pseudovectors = False

    return config


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()

    config.focus_and_target_species_predictor.embedder_config = get_embedder_config()
    config.target_position_predictor.embedder_config = get_embedder_config()

    return config
