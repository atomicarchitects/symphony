"""Defines the default training configuration."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Get the default training configuration."""
    config = ml_collections.ConfigDict()

    # Dataset.
    config.dataset = "platonic_solids"
    config.fragment_logic = "nn"
    config.root_dir = None
    config.shuffle_datasets = True
    config.train_pieces = (None, None)
    config.val_pieces = (None, None)
    config.test_pieces = (None, None)
    config.use_same_seed_for_all_splits = True

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
    config.num_train_steps = 10000
    config.num_eval_steps = 100
    config.num_eval_steps_at_end_of_training = 100
    config.log_every_steps = 500
    config.eval_every_steps = 500
    config.nn_tolerance = 0.1
    config.nn_cutoff = 3.0
    config.compute_padding_dynamically = False
    config.max_n_graphs = 32
    config.max_n_nodes = 15 * config.get_ref("max_n_graphs")
    config.max_n_edges = 45 * config.get_ref("max_n_graphs")
    config.loss_kwargs = ml_collections.ConfigDict()
    config.loss_kwargs.radius_rbf_variance = 1e-3
    config.loss_kwargs.target_position_inverse_temperature = 20.0
    config.loss_kwargs.target_position_lmax = 5
    config.loss_kwargs.ignore_position_loss_for_small_fragments = False
    config.loss_kwargs.position_loss_type = "factorized_kl_divergence"
    config.add_noise_to_positions = False
    config.position_noise_std = 0.0

    # Prediction heads.
    config.focus_and_target_species_predictor = ml_collections.ConfigDict()
    config.focus_and_target_species_predictor.compute_global_embedding = False
    config.focus_and_target_species_predictor.latent_size = 128
    config.focus_and_target_species_predictor.num_layers = 3
    config.focus_and_target_species_predictor.activation = "softplus"

    config.target_position_predictor = ml_collections.ConfigDict()
    config.target_position_predictor.res_beta = 90
    config.target_position_predictor.res_alpha = 179
    config.target_position_predictor.num_channels = 2
    config.target_position_predictor.min_radius = 0.0
    config.target_position_predictor.max_radius = 10.0
    config.target_position_predictor.num_radii = 10
    config.target_position_predictor.apply_gate_on_logits = True
    config.target_position_predictor.square_logits = True
    config.target_position_predictor.target_species_embedding_dim = 32
    config.target_position_predictor.factorized = True
    config.target_position_predictor.num_radial_basis_fns = 8
    config.target_position_predictor.radius_predictor_config = ml_collections.ConfigDict()
    config.target_position_predictor.radius_predictor_config.radius_predictor = "discretized_predictor"
    config.target_position_predictor.radius_predictor_config.min_radius = 0.5
    config.target_position_predictor.radius_predictor_config.max_radius = 2.5
    config.target_position_predictor.radius_predictor_config.num_radii = 5
    config.target_position_predictor.radius_predictor_config.num_layers = 1
    config.target_position_predictor.radius_predictor_config.latent_size = 32
    return config
