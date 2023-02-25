"""Defines the default hyperparameters and training configuration for the MACE model."""

import ml_collections

from configs import default

import e3nn_jax as e3nn


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the MACE model."""
    config = default.get_config()

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    # GNN hyperparameters.
    config.model = "HaikuMACE"
    config.position_coeffs_lmax = 3
    config.output_irreps = str(50 * e3nn.s2_irreps(config.position_coeffs_lmax))
    config.r_max = 5
    config.num_interactions = 1
    config.hidden_irreps = config.output_irreps
    config.readout_mlp_irreps = config.output_irreps
    config.avg_num_neighbors = 15.0
    config.num_species = 5
    config.max_ell = config.position_coeffs_lmax
    return config
