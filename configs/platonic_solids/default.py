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
    config.train_solids = None
    config.val_solids = None
    config.test_solids = None
    config.infer_edges_with_radial_cutoff = True
    config.radial_cutoff = 3.0

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
    config.log_every_steps = 1000
    config.eval_every_steps = 2000
    config.generate_every_steps = 2000
    config.nn_tolerance = 0.1
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
    config.loss_kwargs.radial_loss_scaling_factor = 1.0
    config.loss_kwargs.mask_atom_types = False
    config.mask_atom_types = False
    config.add_noise_to_positions = True
    config.position_noise_std = 0.1

    # Prediction heads.
    config.focus_and_target_species_predictor = ml_collections.ConfigDict()
    config.focus_and_target_species_predictor.compute_global_embedding = False
    config.focus_and_target_species_predictor.global_embedder = (
        ml_collections.ConfigDict()
    )
    config.focus_and_target_species_predictor.global_embedder.num_channels = 1
    config.focus_and_target_species_predictor.global_embedder.pooling = "attention"
    config.focus_and_target_species_predictor.global_embedder.num_attention_heads = 2
    config.focus_and_target_species_predictor.latent_size = 128
    config.focus_and_target_species_predictor.num_layers = 3
    config.focus_and_target_species_predictor.activation = "softplus"

    config.target_position_predictor = ml_collections.ConfigDict()
    config.target_position_predictor.res_beta = 90
    config.target_position_predictor.res_alpha = 179
    config.target_position_predictor.num_channels = 1
    config.target_position_predictor.min_radius = 0.5
    config.target_position_predictor.max_radius = 1.5
    config.target_position_predictor.num_radii = 20
    config.target_position_predictor.apply_gate = False
    config.target_position_predictor.factorized = False
    config.target_position_predictor.radial_mlp_latent_size = 128
    config.target_position_predictor.radial_mlp_num_layers = 2
    config.target_position_predictor.radial_mlp_activation = "swish"

    # Generation.
    config.generation = ml_collections.ConfigDict()
    config.generation.focus_and_atom_type_inverse_temperature = 1.0
    config.generation.position_inverse_temperature = 1.0
    config.generation.res_beta = config.target_position_predictor.get_ref("res_beta")
    config.generation.res_alpha = config.target_position_predictor.get_ref("res_alpha")
    config.generation.radial_cutoff = config.get_ref("radial_cutoff")
    config.generation.num_seeds = 100
    config.generation.num_seeds_per_chunk = 20
    config.generation.init_molecules = "H"
    config.generation.max_num_atoms = 35

    return config
