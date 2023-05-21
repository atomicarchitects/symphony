"""Defines the default training configuration."""

from typing import Optional
import ml_collections
import os


def get_root_dir(dataset: str) -> Optional[str]:
    """Get the root directory for the QM9 dataset."""
    if dataset == "qm9":
        hostname, username = os.uname()[1], os.environ.get("USER")
        if hostname == "radish":
            return "/data/NFS/radish/qm9_fragments/radius"
        if hostname == "potato.mit.edu":
            if username == "songk":
                return "/home/songk/spherical-harmonic-net/qm9_data_tf/data_tf2"
            return "/radish/qm9_fragments/radius"
        elif username == "ameyad":
            return "/Users/ameyad/Documents/qm9_data_tf/radius"
        elif username == "songk":
            return "/Users/songk/atomicarchitects/spherical_harmonic_net/qm9_data_tf/data_tf2"
    return None


def get_config() -> ml_collections.ConfigDict:
    """Get the default training configuration."""
    config = ml_collections.ConfigDict()

    # Dataset.
    config.dataset = "qm9"
    config.train_on_split_smaller_than_chunk = False
    config.root_dir = get_root_dir(config.get_ref("dataset"))
    config.train_molecules = (0, 47616)
    config.val_molecules = (47616, 53568)
    config.test_molecules = (53568, 133920)

    # Optimizer.
    config.optimizer = "adam"
    config.learning_rate = 1e-3
    config.learning_rate_schedule = "constant"
    config.learning_rate_schedule_kwargs = ml_collections.ConfigDict()
    config.learning_rate_schedule_kwargs.init_value = config.get_ref("learning_rate")
    config.learning_rate_schedule_kwargs.peak_value = 2 * config.get_ref(
        "learning_rate"
    )
    config.learning_rate_schedule_kwargs.warmup_steps = 2000
    config.learning_rate_schedule_kwargs.decay_steps = 50000

    # Training.
    config.rng_seed = 0
    config.num_train_steps = 1_000_000
    config.num_eval_steps = 100
    config.num_eval_steps_at_end_of_training = 5000
    config.log_every_steps = 1000
    config.eval_every_steps = 5000
    config.nn_tolerance = 0.5
    config.nn_cutoff = 5.0
    config.compute_padding_dynamically = False
    config.max_n_graphs = 32
    config.max_n_nodes = 30 * config.get_ref("max_n_graphs")
    config.max_n_edges = 90 * config.get_ref("max_n_graphs")
    config.loss_kwargs = ml_collections.ConfigDict()
    config.loss_kwargs.radius_rbf_variance = 1e-3
    config.loss_kwargs.target_position_inverse_temperature = 1.
    config.loss_kwargs.ignore_position_loss_for_small_fragments = False
    config.loss_kwargs.position_loss_type = "kl_divergence"
    config.mask_atom_types = False

    # Prediction heads.
    config.focus_predictor = ml_collections.ConfigDict()
    config.focus_predictor.latent_size = 128
    config.focus_predictor.num_layers = 3

    config.target_species_predictor = ml_collections.ConfigDict()
    config.target_species_predictor.latent_size = 128
    config.target_species_predictor.num_layers = 3

    config.target_position_predictor = ml_collections.ConfigDict()
    config.target_position_predictor.res_beta = 180
    config.target_position_predictor.res_alpha = 359

    return config
