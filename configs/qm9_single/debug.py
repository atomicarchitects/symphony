"""Defines the default hyperparameters and training configuration for the E3SchNet model."""

import ml_collections

from configs.qm9 import e3schnet_and_nequip


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the E3SchNet + NequIP model."""
    config = e3schnet_and_nequip.get_config()

    config.num_train_molecules = 1
    config.num_val_molecules = 1
    config.num_test_molecules = 1
    config.num_train_steps = 10000
    config.num_eval_steps = 10
    config.eval_every_steps = 500
    return config
