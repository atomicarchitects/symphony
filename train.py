"""Library file for executing training and evaluation of generative model."""

import os
from typing import Any, Dict, Iterable, Tuple, Optional, Union

import functools
from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import haiku as hk
import e3nn_jax as e3nn
import flax
import flax.core
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import optax

import input_pipeline
import datatypes
import models


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
    total_loss: metrics.Average.from_output("total_loss")


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
    total_loss: metrics.Average.from_output("total_loss")
    focus_loss: metrics.Average.from_output("focus_loss")
    atom_type_loss: metrics.Average.from_output("atom_type_loss")
    position_loss: metrics.Average.from_output("position_loss")


def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Adds a prefix to the keys of a dict, returning a new dict."""
    return {f"{prefix}_{key}": val for key, val in result.items()}


def create_model(config: ml_collections.ConfigDict) -> nn.Module:
    """Create a model as specified by the config."""
    if config.model == "GraphNet":
        return models.GraphNet(
            latent_size=config.latent_size,
            num_mlp_layers=config.num_mlp_layers,
            message_passing_steps=config.message_passing_steps,
            skip_connections=config.skip_connections,
            layer_norm=config.layer_norm,
            use_edge_model=config.use_edge_model,
        )
    if config.model == "GraphMLP":
        return models.GraphMLP(
            latent_size=config.latent_size,
            num_mlp_layers=config.num_mlp_layers,
            layer_norm=config.layer_norm,
        )
    if config.model == "HaikuGraphMLP":

        def model_fn(graphs):
            return models.HaikuGraphMLP(
                latent_size=config.latent_size,
                num_mlp_layers=config.num_mlp_layers,
                layer_norm=config.layer_norm,
            )(graphs)

        model_fn = hk.transform(model_fn)
        return hk.without_apply_rng(model_fn)
    if config.model == "MACE":
        # TODO (ameyad): Implement MACE in Flax.
        raise NotImplementedError("MACE is not yet implemented.")

    raise ValueError(f"Unsupported model: {config.model}.")


def create_optimizer(config: ml_collections.ConfigDict) -> optax.GradientTransformation:
    """Create an optimizer as specified by the config."""
    if config.optimizer == "adam":
        return optax.adam(learning_rate=config.learning_rate)
    if config.optimizer == "sgd":
        return optax.sgd(learning_rate=config.learning_rate, momentum=config.momentum)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


@functools.partial(jax.jit, static_argnames=["res_beta", "res_alpha"])
def generation_loss(
    preds: datatypes.Predictions,
    graphs: datatypes.Fragment,
    res_beta: int,
    res_alpha: int,
    radius_rbf_variance: float,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the loss for the generation task.
    Args:
        preds (datatypes.Predictions): the model predictions
        graphs (jraph.GraphsTuple): a batch of graphs representing the current molecules
    """
    num_radii = models.RADII.shape[0]
    num_graphs = graphs.n_node.shape[0]
    num_nodes = graphs.nodes.positions.shape[0]
    num_elements = models.NUM_ELEMENTS

    def focus_loss() -> jnp.ndarray:
        # focus_logits is of shape (num_nodes,)
        assert (
            preds.focus_logits.shape
            == graphs.nodes.focus_probability.shape
            == (num_nodes,)
        )

        loss_focus = e3nn.scatter_sum(
            -preds.focus_logits * graphs.nodes.focus_probability, nel=graphs.n_node
        )
        loss_focus += jnp.log(
            1 + e3nn.scatter_sum(jnp.exp(preds.focus_logits), nel=graphs.n_node)
        )
        return loss_focus

    def atom_type_loss() -> jnp.ndarray:
        # species_logits is of shape (num_graphs, num_elements)
        assert (
            preds.species_logits.shape
            == graphs.globals.target_species_probability.shape
            == (num_graphs, num_elements)
        )

        return optax.softmax_cross_entropy(
            logits=preds.species_logits,
            labels=graphs.globals.target_species_probability,
        )

    def position_loss() -> jnp.ndarray:
        # position_coeffs is an e3nn.IrrepsArray of shape (num_graphs, num_radii, dim(irreps))
        assert preds.position_coeffs.array.shape == (
            num_graphs,
            num_radii,
            preds.position_coeffs.irreps.dim,
        )

        # Integrate the position signal over each sphere to get the probability distribution over the radii.
        position_signal = e3nn.to_s2grid(
            preds.position_coeffs,
            res_beta,
            res_alpha,
            quadrature="gausslegendre",
            normalization="integral",
            p_val=1,
            p_arg=-1,
        )

        # position_signal is of shape (num_graphs, num_radii, res_beta, res_alpha)
        assert position_signal.shape == (num_graphs, num_radii, res_beta, res_alpha)

        # For numerical stability, we subtract out the maximum value over each sphere before exponentiating.
        position_max = jnp.max(
            position_signal.grid_values, axis=(-3, -2, -1), keepdims=True
        )

        sphere_normalizing_factors = position_signal.apply(
            lambda pos: jnp.exp(pos - position_max)
        ).integrate()
        sphere_normalizing_factors = sphere_normalizing_factors.array.squeeze(axis=-1)

        # sphere_normalizing_factors is of shape (num_graphs, num_radii)
        assert sphere_normalizing_factors.shape == (
            num_graphs,
            num_radii,
        )

        # position_max is of shape (num_graphs,)
        position_max = position_max.squeeze(axis=(-3, -2, -1))
        assert position_max.shape == (num_graphs,)

        # Compare distance of target relative to focus to target_radius.
        target_positions = graphs.globals.target_positions
        assert target_positions.shape == (num_graphs, 3)

        # Get radius weights from the true distribution, described by a RBF kernel around the target positions.
        radius_weights = jax.vmap(
            lambda target_position: jax.vmap(
                lambda radius: jnp.exp(
                    -((radius - jnp.linalg.norm(target_position)) ** 2)
                    / (2 * radius_rbf_variance)
                )
            )(models.RADII)
        )(target_positions)
        radius_weights = radius_weights / jnp.sum(
            radius_weights, axis=-1, keepdims=True
        )

        # radius_weights is of shape (num_graphs, num_radii)
        assert radius_weights.shape == (num_graphs, num_radii)

        # Compute f(r*, rhat*) which is our model predictions for the target positions.
        target_positions = e3nn.IrrepsArray("1o", target_positions)
        target_positions_logits = jax.vmap(
            functools.partial(e3nn.to_s2point, normalization="integral")
        )(preds.position_coeffs, target_positions)
        target_positions_logits = target_positions_logits.array.squeeze(axis=-1)
        assert target_positions_logits.shape == (num_graphs, num_radii)

        return jax.vmap(
            lambda qr, fr, Zr, c: -jnp.sum(qr * fr) + jnp.log(jnp.sum(Zr)) + c
        )(
            radius_weights,
            target_positions_logits,
            sphere_normalizing_factors,
            position_max,
        )

    loss_focus = focus_loss()
    loss_atom_type = atom_type_loss()
    loss_position = position_loss()

    assert (
        loss_focus.shape == loss_atom_type.shape == loss_position.shape == (num_graphs,)
    )

    total_loss = loss_focus + (loss_atom_type + loss_position) * (
        1 - graphs.globals.stop
    )
    return total_loss, (
        loss_focus,
        loss_atom_type,
        loss_position,
    )


def replace_globals(graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Replaces the globals attribute with a constant feature for each graph."""
    return graphs._replace(
        globals=datatypes.FragmentGlobals(
            stop=jnp.ones([graphs.n_node.shape[0]]),
            target_positions=jnp.ones([graphs.n_node.shape[0], 3]),
            target_species=jnp.ones([graphs.n_node.shape[0]], dtype=jnp.int32),
            target_species_probability=jnp.ones(
                [graphs.n_node.shape[0], models.NUM_ELEMENTS]
            ),
        )
    )


def get_predictions(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
) -> datatypes.Predictions:
    """Get predictions from the network for input graphs."""
    return state.apply_fn(state.params, graphs)


@functools.partial(jax.jit, static_argnames=["loss_kwargs"])
def train_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    loss_kwargs: Dict[str, Union[float, int]],
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params: optax.Params, graphs: jraph.GraphsTuple) -> float:
        curr_state = state.replace(params=params)
        preds = get_predictions(curr_state, graphs)
        loss, *_ = generation_loss(preds=preds, graphs=graphs, **loss_kwargs)
        mask = jraph.get_graph_padding_mask(graphs)
        return jnp.sum(loss * mask) / jnp.sum(mask)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    total_loss, grads = grad_fn(state.params, graphs)
    state = state.apply_gradients(grads=grads)

    metrics_update = TrainMetrics.single_from_model_output(total_loss=total_loss)
    return state, metrics_update


@functools.partial(jax.jit, static_argnames=["loss_kwargs"])
def evaluate_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    loss_kwargs: Dict[str, Union[float, int]],
) -> metrics.Collection:
    """Computes metrics over a set of graphs."""
    # Compute predictions and resulting loss.
    preds = get_predictions(state, graphs)
    total_loss, (focus_loss, atom_type_loss, position_loss) = generation_loss(
        preds=preds, graphs=graphs, **loss_kwargs
    )

    # Take mean over valid graphs.
    mask = jraph.get_graph_padding_mask(graphs)
    total_loss, (focus_loss, atom_type_loss, position_loss) = jax.tree_map(
        lambda arr: jnp.sum(arr * mask) / jnp.sum(mask),
        (total_loss, (focus_loss, atom_type_loss, position_loss)),
    )

    return EvalMetrics.single_from_model_output(
        total_loss=total_loss,
        focus_loss=focus_loss,
        atom_type_loss=atom_type_loss,
        position_loss=position_loss,
    )


def evaluate_model(
    state: train_state.TrainState, datasets, splits: Iterable[str]
) -> Dict[str, metrics.Collection]:
    """Evaluates the model on metrics over the specified splits."""

    # Loop over each split independently.
    eval_metrics = {}
    for split in splits:
        split_metrics = None

        # Loop over graphs.
        for graphs in datasets[split].as_numpy_iterator():
            split_metrics_update = evaluate_step(state, graphs)

            # Update metrics.
            if split_metrics is None:
                split_metrics = split_metrics_update
            else:
                split_metrics = split_metrics.merge(split_metrics_update)
        eval_metrics[split] = split_metrics

    return eval_metrics


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

    # Create writer for logs.
    writer = metric_writers.create_default_writer(workdir)
    writer.write_hparams(config.to_dict())

    # Get datasets, organized by split.
    logging.info("Obtaining datasets.")
    rng = jax.random.PRNGKey(0)
    datasets = input_pipeline.get_datasets(
        rng,
        config.root_dir,
        config.nn_tolerance,
        config.nn_cutoff,
        config.max_n_nodes,
        config.max_n_edges,
        config.max_n_graphs,
    )
    train_iter = datasets["train"]

    # Create and initialize the network.
    logging.info("Initializing network.")
    rng, init_rng = jax.random.split(rng)
    init_graphs = next(train_iter)
    init_graphs = replace_globals(init_graphs)
    net = create_model(config)
    params = jax.jit(net.init)(init_rng, init_graphs)
    parameter_overview.log_parameter_overview(params)

    # Create the optimizer.
    tx = create_optimizer(config)

    # Create the training state.
    state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    # Set up checkpointing of the model.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)
    state = ckpt.restore_or_initialize(state)
    initial_step = int(state.step) + 1

    # Hooks called periodically during training.
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer
    )
    profile = periodic_actions.Profile(
        logdir=workdir, num_profile_steps=5, profile_duration_ms=0
    )
    hooks = [report_progress, profile]

    # Begin training loop.
    logging.info("Starting training.")
    train_metrics = None
    for step in range(initial_step, config.num_train_steps + 1):
        # Perform one step of training.
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            graphs = jax.tree_util.tree_map(np.asarray, next(train_iter))
            state, metrics_update = train_step(
                state,
                graphs,
                loss_kwargs=config.loss_kwargs,
            )

        # Update metrics.
        if train_metrics is None:
            train_metrics = metrics_update
        else:
            train_metrics = train_metrics.merge(metrics_update)

        # Quick indication that training is happening.
        logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)
        for hook in hooks:
            hook(step)

        # Log, if required.
        is_last_step = step == config.num_train_steps
        if step % config.log_every_steps == 0 or is_last_step:
            writer.write_scalars(
                step, add_prefix_to_keys(train_metrics.compute(), "train")
            )
            train_metrics = None

        # # Evaluate on validation and test splits, if required.
        # if step % config.eval_every_steps == 0 or is_last_step:
        #     splits = ["validation", "test"]
        #     with report_progress.timed("eval"):
        #         eval_metrics = evaluate_model(state, datasets, splits=splits)
        #     for split in splits:
        #         writer.write_scalars(
        #             step, add_prefix_to_keys(eval_metrics[split].compute(), split)
        #         )

        # Checkpoint model, if required.
        if step % config.checkpoint_every_steps == 0 or is_last_step:
            with report_progress.timed("checkpoint"):
                ckpt.save(state)

    return state
