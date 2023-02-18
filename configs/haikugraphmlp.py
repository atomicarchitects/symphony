"""Defines the default hyperparameters and training configuration for the GraphMLP model."""

import ml_collections

from . import default


def get_config():
    """Get the hyperparameter configuration for the GraphNetwork model."""
    config = default.get_config()

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    # GNN hyperparameters.
    config.model = "HaikuGraphMLP"
    config.latent_size = 256
    config.num_mlp_layers = 3
    config.layer_norm = True
    return config
