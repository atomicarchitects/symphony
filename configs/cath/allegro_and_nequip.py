"""Defines the default hyperparameters and training configuration for the E3SchNet model."""

import ml_collections

from configs.cath import default, nequip, allegro


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()

    config.focus_and_target_species_predictor.embedder_config = (
        allegro.get_embedder_config()
    )
    config.target_position_predictor.embedder_config = nequip.get_embedder_config()

    return config
