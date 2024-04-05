"""Library file for executing the training and evaluation of generative models."""

import functools
import os
import pickle
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union
import chex
import flax
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import optax
import yaml
from absl import logging
import matplotlib.pyplot as plt

from clu import (
    metric_writers,
    metrics,
    parameter_overview,
    periodic_actions,
)
from flax.training import train_state

from symphony import datatypes, models, loss, helpers
from symphony.data import input_pipeline_tf
from analyses import generate_molecules
from analyses import metrics as analyses_metrics


@flax.struct.dataclass
class Metrics(metrics.Collection):
    total_loss: metrics.Average.from_output("total_loss")
    focus_and_atom_type_loss: metrics.Average.from_output("focus_and_atom_type_loss")
    position_loss: metrics.Average.from_output("position_loss")


class TrainState(train_state.TrainState):
    """State for keeping track of training progress."""

    best_params: flax.core.FrozenDict[str, Any]
    step_for_best_params: float
    metrics_for_best_params: Optional[Dict[str, metrics.Collection]]
    train_metrics: metrics.Collection


def device_batch(
    graph_iterator: Iterator[datatypes.Fragments],
) -> Iterator[datatypes.Fragments]:
    """Batches a set of graphs to the size of the number of devices."""
    num_devices = jax.local_device_count()
    batch = []
    for idx, graph in enumerate(graph_iterator):
        if idx % num_devices == num_devices - 1:
            batch.append(graph)
            batch = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *batch)
            batch = datatypes.Fragments.from_graphstuple(batch)
            yield batch

            batch = []
        else:
            batch.append(graph)


def create_optimizer(config: ml_collections.ConfigDict) -> optax.GradientTransformation:
    """Create an optimizer as specified by the config."""
    # If a learning rate schedule is specified, use it.
    if config.get("learning_rate_schedule") is not None:
        if config.learning_rate_schedule == "constant":
            learning_rate_or_schedule = optax.constant_schedule(config.learning_rate)
        elif config.learning_rate_schedule == "sgdr":
            num_cycles = (
                1
                + config.num_train_steps
                // config.learning_rate_schedule_kwargs.decay_steps
            )
            learning_rate_or_schedule = optax.sgdr_schedule(
                cosine_kwargs=(
                    config.learning_rate_schedule_kwargs for _ in range(num_cycles)
                )
            )
    else:
        learning_rate_or_schedule = config.learning_rate

    if config.optimizer == "adam":
        tx = optax.adam(learning_rate=learning_rate_or_schedule)
    elif config.optimizer == "sgd":
        tx = optax.sgd(
            learning_rate=learning_rate_or_schedule, momentum=config.momentum
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}.")

    if not config.get("freeze_node_embedders"):
        return tx

    # Freeze parameters of the node embedders, if required.
    def flattened_traversal(fn):
        """Returns function that is called with `(path, param)` instead of pytree."""

        def mask(tree):
            flat = flax.traverse_util.flatten_dict(tree)
            return flax.traverse_util.unflatten_dict(
                {k: fn(k, v) for k, v in flat.items()}
            )

        return mask

    # Freezes the node embedders.
    def label_fn(path, param):
        del param
        if path[0].startswith("node_embedder"):
            return "no"
        return "yes"

    return optax.multi_transform(
        {"yes": tx, "no": optax.set_to_zero()}, flattened_traversal(label_fn)
    )


@jax.profiler.annotate_function
def get_predictions(
    state: train_state.TrainState,
    graphs: datatypes.Fragments,
    rng: Optional[chex.Array],
) -> datatypes.Predictions:
    """Get predictions from the network for input graphs."""
    return state.apply_fn(state.params, rng, graphs)


@functools.partial(jax.pmap, axis_name="device", static_broadcasted_argnums=[2, 4, 5])
def train_step(
    graphs: datatypes.Fragments,
    state: train_state.TrainState,
    loss_kwargs: Dict[str, Union[float, int]],
    rng: chex.PRNGKey,
    add_noise_to_positions: bool,
    noise_std: float,
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params: optax.Params, graphs: datatypes.Fragments) -> float:
        curr_state = state.replace(params=params)
        preds = get_predictions(curr_state, graphs, rng=None)
        total_loss, (
            focus_and_atom_type_loss,
            position_loss,
        ) = loss.generation_loss(preds=preds, graphs=graphs, **loss_kwargs)
        mask = jraph.get_graph_padding_mask(graphs)
        mean_loss = jnp.sum(jnp.where(mask, total_loss, 0.0)) / jnp.sum(mask)
        return mean_loss, (
            total_loss,
            focus_and_atom_type_loss,
            position_loss,
            mask,
        )

    # Add noise to positions, if required.
    if add_noise_to_positions:
        noise_rng, rng = jax.random.split(rng)
        position_noise = (
            jax.random.normal(noise_rng, graphs.nodes.positions.shape) * noise_std
        )
    else:
        position_noise = jnp.zeros_like(graphs.nodes.positions)

    noisy_positions = graphs.nodes.positions + position_noise
    graphs = graphs._replace(nodes=graphs.nodes._replace(positions=noisy_positions))

    # Compute gradients.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (
        _,
        (total_loss, focus_and_atom_type_loss, position_loss, mask),
    ), grads = grad_fn(state.params, graphs)

    # Average gradients across devices.
    grads = jax.lax.pmean(grads, axis_name="device")
    state = state.apply_gradients(grads=grads)

    batch_metrics = Metrics.gather_from_model_output(
        axis_name="device",
        total_loss=total_loss,
        focus_and_atom_type_loss=focus_and_atom_type_loss,
        position_loss=position_loss,
        mask=mask,
    )
    return state, batch_metrics


@functools.partial(jax.pmap, axis_name="device", static_broadcasted_argnums=[3])
def evaluate_step(
    graphs: datatypes.Fragments,
    eval_state: train_state.TrainState,
    rng: chex.PRNGKey,
    loss_kwargs: Dict[str, Union[float, int]],
) -> metrics.Collection:
    """Computes metrics over a set of graphs."""
    # Compute predictions and resulting loss.
    preds = get_predictions(eval_state, graphs, rng)
    total_loss, (
        focus_and_atom_type_loss,
        position_loss,
    ) = loss.generation_loss(preds=preds, graphs=graphs, **loss_kwargs)

    # Consider only valid graphs.
    mask = jraph.get_graph_padding_mask(graphs)
    return Metrics.gather_from_model_output(
        axis_name="device",
        total_loss=total_loss,
        focus_and_atom_type_loss=focus_and_atom_type_loss,
        position_loss=position_loss,
        mask=mask,
    )


def evaluate_model(
    eval_state: train_state.TrainState,
    datasets: Dict[str, Iterator[datatypes.Fragments]],
    splits: Iterable[str],
    rng: chex.PRNGKey,
    loss_kwargs: Dict[str, Union[float, int]],
) -> Dict[str, metrics.Collection]:
    """Evaluates the model on metrics over the specified splits."""

    # Loop over each split independently.
    eval_metrics = {}
    for split in splits:
        split_metrics = flax.jax_utils.replicate(Metrics.empty())

        # Loop over graphs.
        for graphs in device_batch(datasets[split].as_numpy_iterator()):
            # Compute metrics for this batch.
            step_rng, rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, jax.local_device_count())
            batch_metrics = evaluate_step(graphs, eval_state, step_rngs, loss_kwargs)
            split_metrics = split_metrics.merge(batch_metrics)

        eval_metrics[split] = flax.jax_utils.unreplicate(split_metrics)

    return eval_metrics


@jax.jit
def mask_atom_types(graphs: datatypes.Fragments) -> datatypes.Fragments:
    """Mask atom types in graphs."""

    def aggregate_sum(arr: jnp.ndarray) -> jnp.ndarray:
        """Aggregates the sum of all elements upto the last in arr into the first element."""
        # Set the first element of arr as the sum of all elements upto the last element.
        # Keep the last element as is.
        # Set all of the other elements to 0.
        return jnp.concatenate(
            [arr[:-1].sum(axis=0, keepdims=True), jnp.zeros_like(arr[:-1]), arr[-1:]],
            axis=0,
        )

    focus_and_target_species_probs = graphs.nodes.focus_and_target_species_probs
    focus_and_target_species_probs = jax.vmap(aggregate_sum)(
        focus_and_target_species_probs
    )
    graphs = graphs._replace(
        nodes=graphs.nodes._replace(
            species=jnp.zeros_like(graphs.nodes.species),
            focus_and_target_species_probs=focus_and_target_species_probs,
        ),
        globals=graphs.globals._replace(
            target_species=jnp.zeros_like(graphs.globals.target_species)
        ),
    )
    return graphs


def train_and_evaluate(
    config: ml_collections.FrozenConfigDict, workdir: str
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the TensorBoard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    # We only support single-host training.
    assert jax.process_count() == 1

    # Check the config file.

    # Create writer for logs.
    writer = metric_writers.create_default_writer(workdir)
    writer.write_hparams(config.to_dict())

    # Save the config for reproducibility.
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, dataset_rng = jax.random.split(rng)
    datasets = input_pipeline_tf.get_datasets(dataset_rng, config)

    # Create and initialize the network.
    logging.info("Initializing network.")
    train_iter = datasets["train"].as_numpy_iterator()
    init_graphs = next(train_iter)
    net = models.create_model(config, run_in_evaluation_mode=False)

    rng, init_rng = jax.random.split(rng)
    params = jax.jit(net.init)(init_rng, init_graphs)
    parameter_overview.log_parameter_overview(params)

    # Create the optimizer.
    tx = create_optimizer(config)

    # Create the training state.
    state = TrainState.create(
        apply_fn=jax.jit(net.apply),
        params=params,
        tx=tx,
        best_params=params,
        step_for_best_params=0,
        metrics_for_best_params=None,
        train_metrics=Metrics.empty(),
    )

    # Create a corresponding evaluation state.
    eval_net = models.create_model(config, run_in_evaluation_mode=False)

    # Set up checkpointing of the model.
    # We will record the best model seen during training.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    checkpoint_hook = helpers.CheckpointHook(checkpoint_dir, max_to_keep=1)
    state = checkpoint_hook.restore_or_initialize(state)

    # Replicate the training and evaluation state across devices.
    state = flax.jax_utils.replicate(state)

    # Hooks called periodically during training.
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer
    )
    profiler = periodic_actions.Profile(
        logdir=workdir,
        every_secs=10800,
    )
    train_metrics_hook = helpers.LogTrainMetricsHook(writer)
    evaluate_model_hook = helpers.EvaluateModelHook(
        evaluate_model_fn=lambda state, rng: evaluate_model(
            state,
            datasets,
            [
                "val_eval",
                "test_eval",
            ],
            rng,
            config.loss_kwargs,
        ),
        writer=writer,
    )
    generate_molecules_hook = helpers.GenerateMoleculesHook(
        workdir=workdir,
        writer=writer,
        focus_and_atom_type_inverse_temperature=config.generation.focus_and_atom_type_inverse_temperature,
        position_inverse_temperature=config.generation.position_inverse_temperature,
        num_seeds=config.generation.num_seeds,
        num_seeds_per_chunk=config.generation.num_seeds_per_chunk,
        init_molecules=config.generation.init_molecules,
        max_num_atoms=config.generation.max_num_atoms,
    )

    # Begin training loop.
    logging.info("Starting training.")

    initial_step = state.step
    for step in range(initial_step, config.num_train_steps):
        # Log, if required.
        first_or_last_step = step in [initial_step, config.num_train_steps]
        if step % config.log_every_steps == 0 or first_or_last_step:
            train_metrics_hook(state)

        # Evaluate model, if required.
        if step % config.eval_every_steps == 0 or first_or_last_step:
            rng, eval_rng = jax.random.split(rng)
            evaluate_model_hook(state, eval_rng)
            checkpoint_hook(state)

        # Generate molecules, if required.
        if step % config.generate_every_steps == 0 or first_or_last_step:
            generate_molecules_hook(step)

        # Get a batch of graphs.
        try:
            graphs = next(device_batch(train_iter))

        except StopIteration:
            logging.info("No more training data. Continuing with final evaluation.")
            break

        # Perform one step of training.
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            step_rng, rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, jax.local_device_count())
            state, batch_metrics = train_step(
                graphs,
                state,
                config.loss_kwargs,
                step_rngs,
                config.add_noise_to_positions,
                config.position_noise_std,
            )

            # Update metrics.
            train_metrics = train_metrics.merge(batch_metrics)

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)
        report_progress(step)
        profiler(step)

    # Once training is complete, return the best state and corresponding metrics.
    logging.info(
        "Evaluating best state from step %d at the end of training.",
        state,
    )
    rng, eval_rng = jax.random.split(rng)
    final_evaluate_model_hook = helpers.EvaluateModelHook(
        evaluate_model_fn=lambda state, rng: evaluate_model(
            state,
            datasets,
            ["val_eval_final", "test_eval_final"],
            rng,
            config.loss_kwargs,
        ),
        writer=writer,
    )
    final_evaluate_model_hook(state, eval_rng)
    checkpoint_hook(state)

    return state
