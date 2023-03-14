"""Defines the default hyperparameters and training configuration for the E3SchNet model."""

import ml_collections
import e3nn_jax as e3nn

from configs import default


def get_config() -> ml_collections.ConfigDict:
    """Get the hyperparameter configuration for the MACE model."""
    config = default.get_config()

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    # GNN hyperparameters.
    config.model = "E3SchNet"
    config.cutoff = 10
    config.n_interactions = 3
    config.n_rbf = 25
    config.n_atom_basis = 128
    config.n_filters = 128
    config.max_ell = 3
    config.activation = "shifted_softplus"
    return config
