import ml_collections

from configs.platonic_solids import default, nequip, e3schnet


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()

    config.focus_and_target_species_predictor.embedder_config = (
        e3schnet.get_embedder_config()
    )
    config.target_position_predictor.embedder_config = nequip.get_embedder_config()
    config.position_updater.embedder_config = nequip.get_embedder_config()

    # NequIP hyperparameters.
    return config