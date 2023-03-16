"""Defines the default hyperparameters and training configuration for the GraphMLP model."""

import ml_collections

from configs import default
import e3nn_jax as e3nn
import jax


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the NequIP model."""
    config = default.get_config()

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    # GNN hyperparameters.
    config.model = "NequIP"
    config.latent_size = 256
    config.layer_norm = True
    config.position_coeffs_lmax = 2

    config.avg_num_neighbors = 15.0
    config.sh_lmax = 3
    config.target_irreps = 64 * e3nn.Irreps("0e + 1o + 2e")
    config.even_activation = jax.nn.swish
    config.odd_activation = jax.nn.tanh
    config.mlp_activation = jax.nn.swish
    config.mlp_n_hidden = 64
    config.mlp_n_layers = 2
    config.n_radial_basis = 8

    return config
