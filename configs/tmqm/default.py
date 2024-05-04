"""Defines the default training configuration."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Get the default training configuration."""
    config = ml_collections.ConfigDict()

    # Dataset.
    config.dataset = "tmqm"
    config.fragment_logic = "nn"
    config.train_on_split_smaller_than_chunk = False
    config.root_dir = None
    config.num_train_molecules = 69000
    config.num_val_molecules = 9000
    config.num_test_molecules = 8665
    config.shuffle_datasets = True
    config.infer_edges_with_radial_cutoff = True
    config.radial_cutoff = 5.0
    config.max_targets_per_graph = 4
    config.heavy_first = False
    config.transition_first = True

    # Optimizer.
    config.optimizer = "adam"
    config.momentum = None
    config.learning_rate = 5e-4
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
    config.use_same_rng_across_structures = False
    config.num_train_steps = 1_000_000
    config.log_every_steps = 1000
    config.eval = True
    config.num_eval_steps = 100
    config.eval_every_steps = 20000
    config.generate = False
    config.generate_every_steps = 2_000_000 #120000
    config.nn_tolerance = 0.5
    config.compute_padding_dynamically = False
    config.max_n_graphs = 16
    config.max_n_nodes = 60 * config.get_ref("max_n_graphs")
    config.max_n_edges = 900 * config.get_ref("max_n_graphs")
    config.loss_kwargs = ml_collections.ConfigDict()
    config.loss_kwargs.ignore_position_loss_for_small_fragments = False
    config.mask_atom_types = False
    config.add_noise_to_positions = True
    config.position_noise_std = 0.05
    config.freeze_node_embedders = False

    # Prediction heads.
    config.focus_and_target_species_predictor = ml_collections.ConfigDict()
    config.focus_and_target_species_predictor.compute_global_embedding = False
    config.focus_and_target_species_predictor.latent_size = 128
    config.focus_and_target_species_predictor.num_layers = 3
    config.focus_and_target_species_predictor.activation = "softplus"

    config.target_position_predictor = ml_collections.ConfigDict()

    config.target_position_predictor.angular_predictor = ml_collections.ConfigDict()
    config.target_position_predictor.angular_predictor.num_channels = 2
    config.target_position_predictor.angular_predictor.apply_gate = False
    config.target_position_predictor.angular_predictor.radial_mlp_num_layers = 2
    config.target_position_predictor.angular_predictor.radial_mlp_latent_size = 8
    config.target_position_predictor.angular_predictor.res_beta = 100
    config.target_position_predictor.angular_predictor.res_alpha = 99
    config.target_position_predictor.angular_predictor.quadrature = "gausslegendre"
    config.target_position_predictor.angular_predictor.sampling_inverse_temperature_factor = 10.0
    config.target_position_predictor.angular_predictor.sampling_num_steps = 1000
    config.target_position_predictor.angular_predictor.sampling_init_step_size = 10.0
    config.target_position_predictor.angular_predictor.continuous = False

    config.target_position_predictor.radial_predictor = ml_collections.ConfigDict()
    config.target_position_predictor.radial_predictor.num_bins = 16
    config.target_position_predictor.radial_predictor.num_param_mlp_layers = 2
    config.target_position_predictor.radial_predictor.num_layers = 2
    config.target_position_predictor.radial_predictor.min_radius = 0.0
    config.target_position_predictor.radial_predictor.max_radius = 5.0
    config.target_position_predictor.radial_predictor.boundary_error = 0.35
    config.target_position_predictor.radial_predictor.latent_size = 128
    config.target_position_predictor.radial_predictor.continuous = False

    # Generation.
    config.generation = ml_collections.ConfigDict()
    config.generation.focus_and_atom_type_inverse_temperature = 1.0
    config.generation.position_inverse_temperature = 1.0
    config.generation.res_beta = config.target_position_predictor.angular_predictor.get_ref("res_beta")
    config.generation.res_alpha = config.target_position_predictor.angular_predictor.get_ref("res_alpha")
    config.generation.radial_cutoff = config.get_ref("radial_cutoff")
    config.generation.num_seeds = 10
    config.generation.num_seeds_per_chunk = 1
    config.generation.init_molecules = "Ni"
    config.generation.max_num_atoms = 200
    config.generation.avg_neighbors_per_atom = 10
    config.generation.species = list(range(1, 81))

    return config
