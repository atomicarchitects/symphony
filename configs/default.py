"""Defines the default training configuration."""

import ml_collections
import os


def get_config() -> ml_collections.ConfigDict:
    """Get the default training configuration."""
    config = ml_collections.ConfigDict()

    hostname = os.uname()[1]
    if hostname == "potato.mit.edu":
        config.root_dir = "/home/ameyad/qm9_data_tf/data_tf2"
    else:
        config.root_dir = "/Users/ameyad/Documents/qm9_data_tf/data_tf2/"

    config.rng_seed = 0
    config.train_molecules = (0, 47616)
    config.val_molecules = (47616, 53568)
    config.test_molecules = (53568, 133920)

    config.num_train_steps = 100_000
    config.num_eval_steps = 100
    config.log_every_steps = 100
    config.eval_every_steps = 500
    config.checkpoint_every_steps = 500
    config.nn_tolerance = 0.5
    config.nn_cutoff = 5.0
    config.max_n_nodes = 512
    config.max_n_edges = 1024
    config.max_n_graphs = 64
    config.loss_kwargs = {
        "res_beta": 30,
        "res_alpha": 51,
        "radius_rbf_variance": 1e-3,
    }

    # Prediction heads.
    config.focus_predictor = ml_collections.ConfigDict()
    config.focus_predictor.latent_size = 128
    config.focus_predictor.num_layers = 2

    config.target_species_predictor = ml_collections.ConfigDict()
    config.target_species_predictor.latent_size = 128
    config.target_species_predictor.num_layers = 2

    config.target_positions_predictor = ml_collections.ConfigDict()
    return config
