"""Defines the default hyperparameters and training configuration for the E3SchNet model."""

import ml_collections

from configs.qm9 import default, nequip


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()
    del config.focus_and_target_species_predictor
    del config.target_position_predictor

    config.position_updater = ml_collections.ConfigDict()
    config.position_updater.embedder_config = nequip.get_embedder_config()

    # NequIP hyperparameters.
    return config
