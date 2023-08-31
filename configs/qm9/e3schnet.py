"""Defines the default hyperparameters and training configuration for the E3SchNet model."""

import ml_collections

from configs.qm9 import default


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the E3SchNet model."""
    config = default.get_config()

    # E3SchNet hyperparameters.
    config.model = "E3SchNet"
    config.cutoff = 5.0
    config.num_interactions = 1
    config.num_basis_fns = 25
    config.num_channels = 32
    config.max_ell = 3
    config.activation = "shifted_softplus"

    return config
