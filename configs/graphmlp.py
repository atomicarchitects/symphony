"""Defines the default hyperparameters and training configuration for the GraphMLP model."""

import ml_collections


def get_config():
    """Get the hyperparameter configuration for the GraphNetwork model."""
    config = ml_collections.ConfigDict()

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1e-3

    # Training hyperparameters.
    config.batch_size = 256
    config.num_train_steps = 100_000
    config.log_every_steps = 100
    config.eval_every_steps = 10_000
    config.checkpoint_every_steps = 10_000
    config.add_virtual_node = True
    config.add_undirected_edges = True
    config.add_self_loops = True
    config.max_n_nodes = 128
    config.max_n_edges = 1024
    config.max_n_graphs = 16

    config.loss_kwargs = {
        "res_beta": 30,
        "res_alpha": 51,
        "radius_rbf_variance": 30,  # what is this
    }

    # GNN hyperparameters.
    config.model = "GraphMLP"
    config.latent_size = 256
    config.dropout_rate = 0.1
    config.num_mlp_layers = 3
    config.layer_norm = True
    return config
