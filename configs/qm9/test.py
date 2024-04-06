"""Defines the default hyperparameters and training configuration for the E3SchNet model."""

import ml_collections

from configs.qm9 import default, nequip, e3schnet


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the E3SchNet + NequIP model."""
    config = default.get_config()

    config.focus_and_target_species_predictor.embedder_config = (
        e3schnet.get_embedder_config()
    )
    config.target_position_predictor.embedder_config = nequip.get_embedder_config()

    config.num_train_steps = 100
    config.num_eval_steps = 10
    config.num_eval_steps_at_end_of_training = 10
    config.eval_every_steps = 500
    config.focus_and_target_species_predictor.max_ell = 5
    config.train_on_split_smaller_than_chunk = True
    config.train_molecules = (0, 1)
    config.use_pseudoscalars_and_pseudovectors = True
    config.add_noise_to_positions = True
    config.position_noise_std = 0.1

    return config
