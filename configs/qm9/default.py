"""Defines the default training configuration."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Get the default training configuration."""
    config = ml_collections.ConfigDict()

    # Dataset.
    config.dataset = "qm9"
    config.fragment_logic = "nn_heavy_first"
    config.train_on_split_smaller_than_chunk = False
    config.root_dir = None
    config.train_molecules = (0, 100000)
    config.val_molecules = (100000, 120000)
    config.test_molecules = (120000, 135000)
    config.shuffle_datasets = True

    # Optimizer.
    config.optimizer = "adam"
    config.momentum = None
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
    config.eval_every_steps = 20000
    config.nn_tolerance = 0.5
    config.nn_cutoff = 5.0
    config.compute_padding_dynamically = False
    config.max_n_graphs = 16
    config.max_n_nodes = 30 * config.get_ref("max_n_graphs")
    config.max_n_edges = 90 * config.get_ref("max_n_graphs")
    config.loss_kwargs = ml_collections.ConfigDict()
    config.loss_kwargs.radius_rbf_variance = 1e-2
    config.loss_kwargs.target_position_inverse_temperature = 20.0
    config.loss_kwargs.target_position_lmax = 5
    config.loss_kwargs.ignore_position_loss_for_small_fragments = False
    config.loss_kwargs.position_loss_type = "kl_divergence"
    config.loss_kwargs.radial_loss_scaling_factor = 1.0
    config.loss_kwargs.mask_atom_types = False
    config.mask_atom_types = False
    config.add_noise_to_positions = True
    config.position_noise_std = 0.1

    # Prediction heads.
    config.focus_and_target_species_predictor = ml_collections.ConfigDict()
    config.focus_and_target_species_predictor.compute_global_embedding = False
    config.focus_and_target_species_predictor.latent_size = 128
    config.focus_and_target_species_predictor.num_layers = 3
    config.focus_and_target_species_predictor.activation = "softplus"

    config.target_position_predictor = ml_collections.ConfigDict()
    config.target_position_predictor.res_beta = 180
    config.target_position_predictor.res_alpha = 359
    config.target_position_predictor.num_channels = 5
    config.target_position_predictor.min_radius = 1.0
    config.target_position_predictor.max_radius = 2.0
    config.target_position_predictor.num_radii = 128
    config.target_position_predictor.apply_gate = False
    config.target_position_predictor.factorized = False
    config.target_position_predictor.radial_mlp_latent_size = 128
    config.target_position_predictor.radial_mlp_num_layers = 2
    config.target_position_predictor.radial_mlp_activation = "swish"

    return config
