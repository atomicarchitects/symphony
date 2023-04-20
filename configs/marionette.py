"""Defines the default hyperparameters and training configuration for the GraphMLP model."""

import ml_collections

from configs import default


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the MarioNette model."""
    config = default.get_config()

    config.model = "MarioNette"
    config.num_channels = 128
    config.r_max = 5.0
    config.avg_num_neighbors = 15.0
    config.num_interactions = 4
    config.max_ell = 5
    config.even_activation = "gelu"
    config.odd_activation = "tanh"
    config.activation = "gelu"
    config.mlp_n_layers = 3
    config.num_basis_fns = 8
    config.soft_normalization = 1e5

    return config
