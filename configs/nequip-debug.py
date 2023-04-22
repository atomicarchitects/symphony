"""Defines the default hyperparameters and training configuration for the GraphMLP model."""

import ml_collections

from configs import default


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()

    # NequIP hyperparameters.
    config.train_molecules = (0, 10)
    config.train_on_split_smaller_than_chunk = True
    config.model = "NequIP"
    config.num_channels = 64
    config.r_max = 5
    config.avg_num_neighbors = 15.0
    config.num_interactions = 4
    config.max_ell = 5
    config.even_activation = "swish"
    config.odd_activation = "tanh"
    config.mlp_activation = "swish"
    config.activation = "softplus"
    config.mlp_n_layers = 2
    config.num_basis_fns = 8

    return config
