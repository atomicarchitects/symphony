"""Library file for executing the training and evaluation of generative models."""

import flax
import jax
import jax.numpy as jnp
import ml_collections
from absl import logging
import haiku as hk

import time
import os
import tensorflow as tf

from clu import (
    checkpoint,
    parameter_overview,
    platform,
)
from flax.training import train_state

from symphony import datatypes, train
from symphony.data import input_pipeline_tf
from symphony.models import utils
from symphony.models.predictor import Predictor
from symphony.models.embedders.global_embedder import GlobalEmbedder
from symphony.models.focus_predictor import FocusAndTargetSpeciesPredictor
from symphony.models.position_predictor import (
    TargetPositionPredictor,
    FactorizedTargetPositionPredictor,
)
from symphony.models.embedders import allegro
import configs.silica.allegro as allegro

ATOMIC_NUMBERS = [8, 14]

import jax.profiler

def create_model(
    config: ml_collections.ConfigDict, run_in_evaluation_mode: bool
) -> hk.Transformed:
    """Create a model as specified by the config."""

    if config.get("position_updater"):
        return utils.create_position_updater(config)

    def model_fn(
        graphs: datatypes.Fragments,
        focus_and_atom_type_inverse_temperature: float = 1.0,
        position_inverse_temperature: float = 1.0,
    ) -> datatypes.Predictions:
        """Defines the entire network."""

        dataset = config.get("dataset", "silica")
        num_species = utils.get_num_species_for_dataset(dataset)

        t = time.time()

        if config.focus_and_target_species_predictor.get("compute_global_embedding"):
            global_embedder = GlobalEmbedder(
                num_channels=config.focus_and_target_species_predictor.global_embedder.num_channels,
                pooling=config.focus_and_target_species_predictor.global_embedder.pooling,
                num_attention_heads=config.focus_and_target_species_predictor.global_embedder.num_attention_heads,
            )
        else:
            global_embedder = None

        print(f'Allegro: {time.time() - t} s')
        t = time.time()

        focus_and_target_species_predictor = FocusAndTargetSpeciesPredictor(
            node_embedder=utils.create_node_embedder(
                config.focus_and_target_species_predictor.embedder_config,
                num_species,
                name_prefix="focus_and_target_species_predictor",
            ),
            global_embedder=global_embedder,
            latent_size=config.focus_and_target_species_predictor.latent_size,
            num_layers=config.focus_and_target_species_predictor.num_layers,
            activation=utils.get_activation(
                config.focus_and_target_species_predictor.activation
            ),
            num_species=num_species,
        )
        print(f'Focus and target species predictor: {time.time() - t} s')
        t = time.time()
        if config.target_position_predictor.get("factorized"):
            target_position_predictor = FactorizedTargetPositionPredictor(
                node_embedder=utils.create_node_embedder(
                    config.target_position_predictor.embedder_config,
                    num_species,
                    name_prefix="target_position_predictor",
                ),
                position_coeffs_lmax=config.target_position_predictor.embedder_config.max_ell,
                res_beta=config.target_position_predictor.res_beta,
                res_alpha=config.target_position_predictor.res_alpha,
                num_channels=config.target_position_predictor.num_channels,
                num_species=num_species,
                min_radius=config.target_position_predictor.min_radius,
                max_radius=config.target_position_predictor.max_radius,
                num_radii=config.target_position_predictor.num_radii,
                radial_mlp_latent_size=config.target_position_predictor.radial_mlp_latent_size,
                radial_mlp_num_layers=config.target_position_predictor.radial_mlp_num_layers,
                radial_mlp_activation=utils.get_activation(
                    config.target_position_predictor.radial_mlp_activation
                ),
                apply_gate=config.target_position_predictor.get("apply_gate"),
            )
        else:
            target_position_predictor = TargetPositionPredictor(
                node_embedder=utils.create_node_embedder(
                    config.target_position_predictor.embedder_config,
                    num_species,
                    name_prefix="target_position_predictor",
                ),
                position_coeffs_lmax=config.target_position_predictor.embedder_config.max_ell,
                res_beta=config.target_position_predictor.res_beta,
                res_alpha=config.target_position_predictor.res_alpha,
                num_channels=config.target_position_predictor.num_channels,
                num_species=num_species,
                min_radius=config.target_position_predictor.min_radius,
                max_radius=config.target_position_predictor.max_radius,
                num_radii=config.target_position_predictor.num_radii,
                apply_gate=config.target_position_predictor.get("apply_gate"),
            )
        print(f'Target position predictor: {time.time() - t} s')
        t = time.time()

        predictor = Predictor(
            focus_and_target_species_predictor=focus_and_target_species_predictor,
            target_position_predictor=target_position_predictor,
        )

        return get_training_predictions(predictor, graphs)

    return hk.transform(model_fn)


def get_training_predictions(
    predictor, graphs: datatypes.Fragments
) -> datatypes.Predictions:
    """Returns the predictions on these graphs during training, when we have access to the true focus and target species."""
    print("get predictions")
    t = time.time()
    # Get the number of graphs and nodes.
    num_nodes = graphs.nodes.positions.shape[0]
    num_graphs = graphs.n_node.shape[0]
    num_species = predictor.focus_and_target_species_predictor.num_species
    segment_ids = utils.get_segment_ids(graphs.n_node, num_nodes)

    print(f'Get segment ids: {time.time() - t} s')
    t = time.time()

    # Get the species and stop logits.
    (
        focus_and_target_species_logits,
        stop_logits,
    ) = predictor.focus_and_target_species_predictor(graphs)

    print(f'Get species/stop logits: {time.time() - t} s')
    t = time.time()

    # Get the species and stop probabilities.
    focus_and_target_species_probs, stop_probs = utils.segment_softmax_2D_with_stop(
        focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
    )

    print(f'Get species/stop probs: {time.time() - t} s')
    t = time.time()

    # Get the embeddings of the focus nodes.
    # These are the first nodes in each graph during training.
    focus_node_indices = utils.get_first_node_indices(graphs)

    print(f'Get focus node embeddings: {time.time() - t} s')
    t = time.time()

    # Get the coefficients for the target positions.
    (
        log_position_coeffs,
        position_logits,
        angular_logits,
        radial_logits,
    ) = predictor.target_position_predictor(
        graphs,
        focus_node_indices,
        graphs.globals.target_species,
        inverse_temperature=1.0,
    )
    print(f"Get target position coeffs: {time.time() - t} s")
    t = time.time()

    # Get the position probabilities.
    position_probs = jax.vmap(utils.position_logits_to_position_distribution)(
        position_logits
    )
    print(f"Get position probs: {time.time() - t} s")
    t = time.time()

    # The radii bins used for the position prediction, repeated for each graph.
    radii = predictor.target_position_predictor.create_radii()
    radial_bins = jax.vmap(lambda _: radii)(jnp.arange(num_graphs))

    # Check the shapes.
    assert focus_and_target_species_logits.shape == (
        num_nodes,
        num_species,
    )
    assert focus_and_target_species_probs.shape == (
        num_nodes,
        num_species,
    )
    assert log_position_coeffs.shape == (
        num_graphs,
        predictor.target_position_predictor.num_channels,
        predictor.target_position_predictor.num_radii,
        log_position_coeffs.shape[-1],
    )
    assert position_logits.shape == (
        num_graphs,
        predictor.target_position_predictor.num_radii,
        predictor.target_position_predictor.res_beta,
        predictor.target_position_predictor.res_alpha,
    )

    return datatypes.Predictions(
        nodes=datatypes.NodePredictions(
            focus_and_target_species_logits=focus_and_target_species_logits,
            focus_and_target_species_probs=focus_and_target_species_probs,
            embeddings_for_focus=predictor.focus_and_target_species_predictor.compute_node_embeddings(
                graphs
            ),
            embeddings_for_positions=predictor.target_position_predictor.compute_node_embeddings(
                graphs
            ),
        ),
        edges=None,
        globals=datatypes.GlobalPredictions(
            stop_logits=stop_logits,
            stop_probs=stop_probs,
            stop=None,
            focus_indices=focus_node_indices,
            target_species=None,
            log_position_coeffs=log_position_coeffs,
            position_logits=position_logits,
            position_probs=position_probs,
            position_vectors=None,
            radial_bins=radial_bins,
            radial_logits=radial_logits,
            angular_logits=angular_logits,
        ),
        senders=graphs.senders,
        receivers=graphs.receivers,
        n_node=graphs.n_node,
        n_edge=graphs.n_edge,
    )


# Make sure the dataloader is deterministic.
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

# Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.experimental.set_visible_devices([], "GPU")

# We only support single-host training on a single device.
logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
logging.info("JAX local devices: %r", jax.local_devices())
logging.info("CUDA_VISIBLE_DEVICES: %r", os.environ.get("CUDA_VISIBLE_DEVICES"))

# Add a note so that we can tell which task is which JAX host.
# (Depending on the platform task 0 is not guaranteed to be host 0)
platform.work_unit().set_task_status(
    f"process_index: {jax.process_index()}, "
    f"process_count: {jax.process_count()}"
)

config = allegro.get_config()
config = ml_collections.FrozenConfigDict(config)

# Get datasets, organized by split.
logging.info("Obtaining datasets.")
rng = jax.random.PRNGKey(config.rng_seed)
rng, dataset_rng = jax.random.split(rng)
datasets = input_pipeline_tf.get_datasets(dataset_rng, config)
# Create and initialize the network.
logging.info("Initializing network.")
train_iter = datasets["train"].as_numpy_iterator()
init_graphs = next(train_iter)
net = create_model(config, run_in_evaluation_mode=False)

rng, init_rng = jax.random.split(rng)
params = jax.jit(net.init)(init_rng, init_graphs)
parameter_overview.log_parameter_overview(params)

# Create the optimizer.
tx = train.create_optimizer(config)

# Create the training state.
state = train_state.TrainState.create(
    apply_fn=jax.jit(net.apply), params=params, tx=tx
)

# Set up checkpointing of the model.
# We will record the best model seen during training.
ckpt = checkpoint.Checkpoint(".", max_to_keep=5)
restored = ckpt.restore_or_initialize(
    {
        "state": state,
        "best_state": state,
        "step_for_best_state": 1.0,
        "metrics_for_best_state": None,
    }
)
state = restored["state"]
best_state = restored["best_state"]
step_for_best_state = restored["step_for_best_state"]
metrics_for_best_state = restored["metrics_for_best_state"]
if metrics_for_best_state is None:
    min_val_loss = float("inf")
else:
    min_val_loss = metrics_for_best_state["val_eval"]["total_loss"]
initial_step = int(state.step) + 1

# Replicate the training and evaluation state across devices.
state = flax.jax_utils.replicate(state)
best_state = flax.jax_utils.replicate(best_state)


# Begin training loop.
logging.info("Starting training.")
train_metrics = flax.jax_utils.replicate(train.Metrics.empty())
train_metrics_empty = True

step = initial_step

with jax.profiler.trace("/home/songk/tensorboard", create_perfetto_trace=True):
    # Log, if required.
    first_or_last_step = step in [initial_step, config.num_train_steps]
    if step % config.log_every_steps == 0 or first_or_last_step:
        train_metrics = flax.jax_utils.replicate(train.Metrics.empty())
        train_metrics_empty = True

    # Get a batch of graphs.
    graphs = next(train.device_batch(train_iter))

    # Perform one step of training.
    step_rng, rng = jax.random.split(rng)
    step_rngs = jax.random.split(step_rng, jax.local_device_count())
    state, batch_metrics = train.train_step(
        graphs,
        state,
        config.loss_kwargs,
        step_rngs,
        config.add_noise_to_positions,
        config.position_noise_std,
    )

    t = time.time()

    # Update metrics.
    train_metrics = train_metrics.merge(batch_metrics)
    train_metrics_empty = False
    print(f"Update metrics: {time.time() - t} s")
    jax.block_until_ready(train_metrics)

print("done")
