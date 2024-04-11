
import ml_collections

from configs.platonic_solids import nequip


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the E3SchNet + NequIP model."""
    config = nequip.get_config()

    # Modify the default configuration, just to test.
    config.num_train_steps = 100
    config.num_eval_steps = 10
    config.num_eval_steps_at_end_of_training = 10
    config.eval_every_steps = 500
    config.focus_and_target_species_predictor.max_ell = 5
    config.use_pseudoscalars_and_pseudovectors = True
    config.add_noise_to_positions = True
    config.position_noise_std = 0.1

    return config

