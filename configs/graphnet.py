"""Defines the default hyperparameters and training configuration for the GraphNetwork model."""

import ml_collections

from configs import default


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the GraphNetwork model."""
    config = default.get_config()

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    # GNN hyperparameters.
    config.model = "GraphNet"
    config.message_passing_steps = 5
    config.latent_size = 256
    config.num_mlp_layers = 1
    config.use_edge_model = True
    config.skip_connections = True
    config.layer_norm = True
    config.position_coeffs_lmax = 2
    return config
