"""Defines the default hyperparameters and training configuration for the NequIP model."""

import ml_collections

from configs import nequip


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = nequip.get_config()

    # NequIP hyperparameters.
    config.train_on_split_smaller_than_chunk = True
    return config
