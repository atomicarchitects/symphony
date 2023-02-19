"""Defines the default hyperparameters and training configuration for the MACE model."""

import ml_collections

from . import default
import e3nn_jax as e3nn


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

    config.output_irreps = "128x0e + 32x1o + 32x2e + 32x3o + 32x4e + 32x5o"
    config.r_max = 5
    config.num_interactions = 2
    config.hidden_irreps = "128x0e + 128x1o + 128x2e"
    config.readout_mlp_irreps = "128x0e + 128x1o + 128x2e"
    config.avg_num_neighbors = 3
    config.num_species = 5
    config.max_ell = 3
    return config
