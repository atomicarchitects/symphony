"""Defines the default training configuration."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Get the default training configuration."""
    config = ml_collections.ConfigDict()

    config.root_dir = "qm9_data"
    config.batch_size = 256
    config.num_train_steps = 100_000
    config.log_every_steps = 100
    config.eval_every_steps = 10_000
    config.checkpoint_every_steps = 10_000
    config.nn_tolerance = 0.5
    config.nn_cutoff = 5.0
    config.num_train_graphs = 50_000
    config.num_val_graphs = 5_000
    config.num_test_graphs = 80_000
    config.max_n_nodes = 128
    config.max_n_edges = 1024
    config.max_n_graphs = 16
    config.loss_kwargs = {
        "res_beta": 30,
        "res_alpha": 51,
        "radius_rbf_variance": 1e-3,
    }
    return config
