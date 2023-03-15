"""Library file for executing training and evaluation of generative model."""

import functools
import os
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import chex
import e3nn_jax as e3nn
import flax
import flax.core
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import optax
import yaml
from absl import logging
from clu import (
    checkpoint,
    metric_writers,
    metrics,
    parameter_overview,
    periodic_actions,
)
from flax.training import train_state

import datatypes
import input_pipeline_tf
import models


@flax.struct.dataclass
class TrainMetrics(metrics.Collection):
    total_loss: metrics.Average.from_output("total_loss")
    focus_loss: metrics.Average.from_output("focus_loss")
    atom_type_loss: metrics.Average.from_output("atom_type_loss")
    position_loss: metrics.Average.from_output("position_loss")


@flax.struct.dataclass
class EvalMetrics(metrics.Collection):
    total_loss: metrics.Average.from_output("total_loss")
    focus_loss: metrics.Average.from_output("focus_loss")
    atom_type_loss: metrics.Average.from_output("atom_type_loss")
    position_loss: metrics.Average.from_output("position_loss")


def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Adds a prefix to the keys of a dict, returning a new dict."""
    return {f"{prefix}_{key}": val for key, val in result.items()}


def create_model(
    config: ml_collections.ConfigDict, run_in_evaluation_mode: bool
) -> nn.Module:
    """Create a model as specified by the config."""

    def model_fn(graphs):
        if config.activation == "shifted_softplus":
            activation = models.shifted_softplus
        else:
            activation = getattr(jax.nn, config.activation)

        if config.model == "MACE":
            output_irreps = config.num_channels * e3nn.s2_irreps(config.max_ell)

            node_embedder = models.MACE(
                output_irreps=output_irreps,
                hidden_irreps=output_irreps,
                readout_mlp_irreps=output_irreps,
                r_max=config.r_max,
                num_interactions=config.num_interactions,
                avg_num_neighbors=config.avg_num_neighbors,
                num_species=config.num_species,
                max_ell=config.max_ell,
                num_basis_fns=config.num_basis_fns,
            )
        elif config.model == "E3SchNet":
            node_embedder = models.E3SchNet(
                n_atom_basis=config.n_atom_basis,
                n_interactions=config.n_interactions,
                n_filters=config.n_filters,
                n_rbf=config.n_rbf,
                activation=activation,
                cutoff=config.cutoff,
                max_ell=config.max_ell,
            )
        else:
            raise ValueError(f"Unsupported model: {config.model}.")

        focus_predictor = models.FocusPredictor(
            latent_size=config.focus_predictor.latent_size,
            num_layers=config.focus_predictor.num_layers,
            activation=activation,
        )
        target_species_predictor = models.TargetSpeciesPredictor(
            latent_size=config.target_species_predictor.latent_size,
            num_layers=config.target_species_predictor.num_layers,
            activation=activation,
        )
        target_position_predictor = models.TargetPositionPredictor(
            position_coeffs_lmax=config.max_ell
        )
        return models.Predictor(
            node_embedder=node_embedder,
            focus_predictor=focus_predictor,
            target_species_predictor=target_species_predictor,
            target_position_predictor=target_position_predictor,
            run_in_evaluation_mode=run_in_evaluation_mode,
        )(graphs)

    return hk.transform(model_fn)


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

        n_node = graphs.n_node
        focus_logits = preds.focus_logits

        # Compute sum(qv * fv) for each graph, where fv is the focus_logits for node v.
        loss_focus = e3nn.scatter_sum(
            -graphs.nodes.focus_probability * focus_logits, nel=n_node
        )

        # This is basically log(1 + sum(exp(fv))) for each graph.
        # But we subtract out the maximum fv per graph for numerical stability.
        focus_logits_max = e3nn.scatter_max(focus_logits, nel=n_node, initial=0.0)
        focus_logits_max_expanded = e3nn.scatter_max(
            focus_logits, nel=n_node, map_back=True, initial=0.0
        )
        focus_logits -= focus_logits_max_expanded
        loss_focus += focus_logits_max + jnp.log(
            jnp.exp(-focus_logits_max)
            + e3nn.scatter_sum(jnp.exp(focus_logits), nel=n_node)
        )

        assert loss_focus.shape == (num_graphs,)
        return loss_focus

    def atom_type_loss() -> jnp.ndarray:
        # target_species_logits is of shape (num_graphs, num_elements)
        assert (
            preds.target_species_logits.shape
            == graphs.globals.target_species_probability.shape
            == (num_graphs, num_elements)
        )

        loss_atom_type = optax.softmax_cross_entropy(
            logits=preds.target_species_logits,
            labels=graphs.globals.target_species_probability,
        )

        assert loss_atom_type.shape == (num_graphs,)
        return loss_atom_type

    def position_loss() -> jnp.ndarray:
        # position_coeffs is an e3nn.IrrepsArray of shape (num_graphs, num_radii, dim(irreps))
        assert preds.position_coeffs.array.shape == (
            num_graphs,
            num_radii,
            preds.position_coeffs.irreps.dim,
        )

        # Compute the position signal projected to a spherical grid for each radius.
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

        # Integrate the position signal over each sphere to get the normalizing factors for the radii.
        # For numerical stability, we subtract out the maximum value over all spheres before exponentiating.
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

        # These are the target positions for each graph.
        # We need to compare these predicted positions to the target positions.
        target_positions = graphs.globals.target_positions
        assert target_positions.shape == (num_graphs, 3)

        # Get radius weights from the true distribution,
        # described by a RBF kernel around the target positions.
        true_radius_weights = jax.vmap(
            lambda target_position: jax.vmap(
                lambda radius: jnp.exp(
                    -((radius - jnp.linalg.norm(target_position)) ** 2)
                    / (2 * radius_rbf_variance)
                )
            )(models.RADII)
        )(target_positions)

        # Normalize to get a probability distribution.
        true_radius_weights += 1e-10
        true_radius_weights = true_radius_weights / jnp.sum(
            true_radius_weights, axis=-1, keepdims=True
        )

        # true_radius_weights is of shape (num_graphs, num_radii)
        assert true_radius_weights.shape == (num_graphs, num_radii)

        # Compute f(r*, rhat*) which is our model predictions for the target positions.
        target_positions = e3nn.IrrepsArray("1o", target_positions)
        target_positions_logits = jax.vmap(
            functools.partial(e3nn.to_s2point, normalization="integral")
        )(preds.position_coeffs, target_positions)
        target_positions_logits = target_positions_logits.array.squeeze(axis=-1)
        assert target_positions_logits.shape == (num_graphs, num_radii)

        loss_position = jax.vmap(
            lambda qr, fr, Zr, c: -jnp.sum(qr * fr) + jnp.log(jnp.sum(Zr)) + c
        )(
            true_radius_weights,
            target_positions_logits,
            sphere_normalizing_factors,
            position_max,
        )

        assert loss_position.shape == (num_graphs,)
        return loss_position

    # If this is the last step in the generation process, we do not have to predict atom type and position.
    loss_focus = focus_loss()
    loss_atom_type = atom_type_loss() * (1 - graphs.globals.stop)
    loss_position = position_loss() * (1 - graphs.globals.stop)

    total_loss = loss_focus + loss_atom_type + loss_position
    return total_loss, (
        loss_focus,
        loss_atom_type,
        loss_position,
    )


def get_predictions(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    rng: Optional[chex.Array],
) -> datatypes.Predictions:
    """Get predictions from the network for input graphs."""
    return state.apply_fn(state.params, rng, graphs)


@functools.partial(jax.jit, static_argnames=["loss_kwargs"])
def train_step(
    state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    loss_kwargs: Dict[str, Union[float, int]],
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params: optax.Params, graphs: jraph.GraphsTuple) -> float:
        curr_state = state.replace(params=params)
        preds = get_predictions(curr_state, graphs, rng=None)
        total_loss, (focus_loss, atom_type_loss, position_loss) = generation_loss(
            preds=preds, graphs=graphs, **loss_kwargs
        )
        mask = jraph.get_graph_padding_mask(graphs)
        mean_loss = jnp.sum(jnp.where(mask, total_loss, 0.0)) / jnp.sum(mask)
        return mean_loss, (total_loss, focus_loss, atom_type_loss, position_loss, mask)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (
        _,
        (total_loss, focus_loss, atom_type_loss, position_loss, mask),
    ), grads = grad_fn(state.params, graphs)
    state = state.apply_gradients(grads=grads)

    batch_metrics = TrainMetrics.single_from_model_output(
        total_loss=total_loss,
        focus_loss=focus_loss,
        atom_type_loss=atom_type_loss,
        position_loss=position_loss,
        mask=mask,
    )
    return state, batch_metrics


@functools.partial(jax.jit, static_argnames=["loss_kwargs"])
def evaluate_step(
    eval_state: train_state.TrainState,
    graphs: jraph.GraphsTuple,
    rng: chex.PRNGKey,
    loss_kwargs: Dict[str, Union[float, int]],
) -> metrics.Collection:
    """Computes metrics over a set of graphs."""
    # Compute predictions and resulting loss.
    preds = get_predictions(eval_state, graphs, rng)
    total_loss, (focus_loss, atom_type_loss, position_loss) = generation_loss(
        preds=preds, graphs=graphs, **loss_kwargs
    )

    # Consider only valid graphs.
    mask = jraph.get_graph_padding_mask(graphs)
    return EvalMetrics.single_from_model_output(
        total_loss=total_loss,
        focus_loss=focus_loss,
        atom_type_loss=atom_type_loss,
        position_loss=position_loss,
        mask=mask,
    )


def evaluate_model(
    eval_state: train_state.TrainState,
    datasets: Iterator[datatypes.Fragment],
    splits: Iterable[str],
    rng: chex.PRNGKey,
    loss_kwargs: Dict[str, Union[float, int]],
    num_eval_steps: int,
) -> Dict[str, metrics.Collection]:
    """Evaluates the model on metrics over the specified splits."""

    # Loop over each split independently.
    eval_metrics = {}
    for split in splits:
        split_metrics = None

        # Loop over graphs.
        for graphs in datasets[split].take(num_eval_steps).as_numpy_iterator():
            graphs = datatypes.Fragment.from_graphstuple(graphs)

            # Compute metrics for this batch.
            step_rng, rng = jax.random.split(rng)
            batch_metrics = evaluate_step(eval_state, graphs, step_rng, loss_kwargs)

            # Update metrics.
            if split_metrics is None:
                split_metrics = batch_metrics
            else:
                split_metrics = split_metrics.merge(batch_metrics)
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
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, dataset_rng = jax.random.split(rng)
    # datasets = input_pipeline.get_datasets(dataset_rng, config)
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
    tx = create_optimizer(config)

    # Create the training state.
    state = train_state.TrainState.create(
        apply_fn=jax.jit(net.apply), params=params, tx=tx
    )

    # Create a corresponding evaluation state.
    eval_net = create_model(config, run_in_evaluation_mode=True)
    eval_state = state.replace(apply_fn=jax.jit(eval_net.apply))

    # Set up checkpointing of the model.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=5)
    state = ckpt.restore_or_initialize(state)
    initial_step = int(state.step) + 1

    # Save the config for reproducibility.
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Hooks called periodically during training.
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_train_steps, writer=writer
    )
    profile = periodic_actions.Profile(
        logdir=workdir,
        num_profile_steps=5,
    )
    hooks = [report_progress, profile]

    # We will record the best model seen during training.
    best_state = state
    min_val_loss = jnp.inf

    # Begin training loop.
    logging.info("Starting training.")
    train_metrics = None
    for step in range(initial_step, config.num_train_steps + 1):
        # Perform one step of training.
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            graphs = next(train_iter)
            graphs = datatypes.Fragment.from_graphstuple(graphs)
            state, batch_metrics = train_step(
                state,
                graphs,
                loss_kwargs=config.loss_kwargs,
            )

        # Update metrics.
        if train_metrics is None:
            train_metrics = batch_metrics
        else:
            train_metrics = train_metrics.merge(batch_metrics)

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

        # Evaluate on validation and test splits, if required.
        if step % config.eval_every_steps == 0 or is_last_step:
            # Evaluate with the current set of parameters.
            eval_state = eval_state.replace(params=state.params)

            splits = ["val", "test"]
            with report_progress.timed("eval"):
                rng, eval_rng = jax.random.split(rng)
                eval_metrics = evaluate_model(
                    eval_state,
                    datasets,
                    splits,
                    eval_rng,
                    config.loss_kwargs,
                    config.num_eval_steps,
                )
            for split in splits:
                eval_metrics[split] = eval_metrics[split].compute()
                writer.write_scalars(
                    step, add_prefix_to_keys(eval_metrics[split], split)
                )

            # Note best state seen so far.
            # Best state is defined as the state with the lowest validation loss.
            if eval_metrics["val"]["total_loss"] < min_val_loss:
                min_val_loss = eval_metrics["val"]["total_loss"]
                best_state = state
                metrics_for_best_state = eval_metrics

                # Checkpoint the best state and corresponding metrics seen during training.
                with report_progress.timed("checkpoint"):
                    ckpt.save(
                        {
                            "best_state": best_state,
                            "metrics_for_best_state": metrics_for_best_state,
                        }
                    )

    # Checkpoint the best state and corresponding metrics seen after training is complete.
    with report_progress.timed("checkpoint"):
        ckpt.save(
            {
                "best_state": best_state,
                "metrics_for_best_state": metrics_for_best_state,
            }
        )

    return best_state
