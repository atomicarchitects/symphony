"""Defines the default hyperparameters and training configuration for the MACE model."""

import ml_collections

from configs import default


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the MACE model."""
    config = default.get_config()

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    # MACE hyperparameters.
    config.model = "MACE"
    config.num_channels = 128
    config.r_max = 5
    config.num_interactions = 1
    config.avg_num_neighbors = 15.0
    config.num_species = 5
    config.max_ell = 3
    config.num_basis_fns = 8
    config.activation = "softplus"

    return ml_collections.FrozenConfigDict(config)
