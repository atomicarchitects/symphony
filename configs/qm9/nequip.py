"""Defines the default hyperparameters and training configuration for the NequIP model."""

import ml_collections

from configs.qm9 import default


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()

    # NequIP hyperparameters.
    config.model = "NequIP"
    config.num_channels = 64
    config.r_max = 5
    config.avg_num_neighbors = 300.0  # NequIP is not properly normalized.
    config.num_interactions = 4
    config.max_ell = 5
    config.even_activation = "swish"
    config.odd_activation = "tanh"
    config.mlp_activation = "swish"
    config.activation = "softplus"
    config.mlp_n_layers = 2
    config.num_basis_fns = 8
    config.skip_connection = True

    return config
