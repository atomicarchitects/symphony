"""Defines the default hyperparameters and training configuration for the NequIP model."""

import ml_collections

from configs.qm9 import default


def get_embedder_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.model = "NequIP"
    config.num_channels = 64
    config.r_max = 5
    config.avg_num_neighbors = 300.0  # NequIP is not properly normalized.
    config.num_interactions = 3
    config.max_ell = 3
    config.even_activation = "swish"
    config.odd_activation = "tanh"
    config.mlp_activation = "swish"
    config.activation = "softplus"
    config.mlp_n_layers = 2
    config.num_basis_fns = 8
    config.skip_connection = True
    config.use_pseudoscalars_and_pseudovectors = False
    return config


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()

    config.focus_and_target_species_predictor.embedder_config = get_embedder_config()
    config.target_position_predictor.embedder_config = get_embedder_config()

    # NequIP hyperparameters.
    return config
