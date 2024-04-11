
import ml_collections

from configs.platonic_solids import e3schnet_and_nequip


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the E3SchNet + NequIP model."""
    config = e3schnet_and_nequip.get_config()

    # Modify the default configuration, just to test.
    config.num_train_steps = 100
    config.num_eval_steps = 10
    config.eval_every_steps = 500
    config.focus_and_target_species_predictor.embedder_config.max_ell = 1
    config.target_position_predictor.embedder_config.max_ell = 1
    config.add_noise_to_positions = True
    config.position_noise_std = 0.1

    return config

