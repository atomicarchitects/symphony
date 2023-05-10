"""Library file for executing training and evaluation of generative model."""

import functools
import os
import pickle
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

import chex
import e3nn_jax as e3nn
import flax
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
from models import create_model


@flax.struct.dataclass
class Metrics(metrics.Collection):
    total_loss: metrics.Average.from_output("total_loss")
    focus_loss: metrics.Average.from_output("focus_loss")
    atom_type_loss: metrics.Average.from_output("atom_type_loss")
    position_loss: metrics.Average.from_output("position_loss")


def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Adds a prefix to the keys of a dict, returning a new dict."""
    return {f"{prefix}/{key}": val for key, val in result.items()}


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
        return optax.adam(learning_rate=learning_rate_or_schedule)
    if config.optimizer == "sgd":
        return optax.sgd(
            learning_rate=learning_rate_or_schedule, momentum=config.momentum
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}.")


@functools.partial(jax.profiler.annotate_function, name="generation_loss")
def generation_loss(
    preds: datatypes.Predictions,
    graphs: datatypes.Fragments,
    radius_rbf_variance: float,
    target_position_inverse_temperature: float,
    ignore_position_loss_for_small_fragments: bool,
    position_loss_type: str,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Computes the loss for the generation task.
    Args:
        preds (datatypes.Predictions): the model predictions
        graphs (datatypes.Fragment): a batch of graphs representing the current molecules
    """
    num_radii = models.RADII.shape[0]
    num_graphs = graphs.n_node.shape[0]
    num_nodes = graphs.nodes.positions.shape[0]
    num_elements = models.NUM_ELEMENTS

    def focus_loss() -> jnp.ndarray:
        """Computes the loss over focus probabilities."""
        assert (
            preds.nodes.focus_logits.shape
            == graphs.nodes.focus_probability.shape
            == (num_nodes,)
        )

        n_node = graphs.n_node
        focus_logits = preds.nodes.focus_logits

        # Compute sum(qv * fv) for each graph,
        # where fv is the focus_logits for node v,
        # and qv is the focus_probability for node v.
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
        """Computes the loss over atom types."""
        assert (
            preds.globals.target_species_logits.shape
            == graphs.globals.target_species_probability.shape
            == (num_graphs, num_elements)
        )

        loss_atom_type = optax.softmax_cross_entropy(
            logits=preds.globals.target_species_logits,
            labels=graphs.globals.target_species_probability,
        )

        assert loss_atom_type.shape == (num_graphs,)
        return loss_atom_type

    def target_position_to_radius_weights(
        target_position: jnp.ndarray,
    ) -> jnp.ndarray:
        """Returns the radial distribution for a target position."""
        radius_weights = jax.vmap(
            lambda radius: jnp.exp(
                -((radius - jnp.linalg.norm(target_position)) ** 2)
                / (2 * radius_rbf_variance)
            )
        )(models.RADII)
        radius_weights += 1e-10
        return radius_weights / jnp.sum(radius_weights)

    def position_loss_with_kl_divergence() -> jnp.ndarray:
        """Computes the loss over position probabilities using the KL divergence."""

        position_logits = preds.globals.position_logits
        res_beta, res_alpha, quadrature = (
            position_logits.res_beta,
            position_logits.res_alpha,
            position_logits.quadrature,
        )

        def safe_log(x: jnp.ndarray) -> jnp.ndarray:
            """Computes the log of x, replacing 0 with 1 for numerical stability."""
            return jnp.log(jnp.where(x == 0, 1.0, x))

        def target_position_to_log_angular_coeffs(
            target_position: jnp.ndarray,
        ) -> e3nn.IrrepsArray:
            """Returns the temperature-scaled angular distribution for a target position."""
            # Compute the true distribution over positions,
            # described by a smooth approximation of a delta function at the target positions.
            norm = jnp.linalg.norm(target_position, axis=-1, keepdims=True)
            target_position_unit_vector = target_position / jnp.where(
                norm == 0, 1, norm
            )
            target_position_unit_vector = e3nn.IrrepsArray(
                "1o", target_position_unit_vector
            )
            return target_position_inverse_temperature * target_position_unit_vector

        def kl_divergence_on_spheres(
            true_radius_weights: jnp.ndarray,
            log_true_angular_coeffs: e3nn.IrrepsArray,
            log_predicted_dist: e3nn.SphericalSignal,
        ) -> jnp.ndarray:
            """Compute the KL divergence between two distributions on the spheres."""
            # Convert coefficients to a distribution on the sphere.
            log_true_angular_dist = e3nn.to_s2grid(
                log_true_angular_coeffs,
                res_beta,
                res_alpha,
                quadrature=quadrature,
                p_val=1,
                p_arg=-1,
            )

            # Subtract the maximum value for numerical stability.
            log_true_angular_dist_max = jnp.max(
                log_true_angular_dist.grid_values, axis=(-2, -1), keepdims=True
            )
            log_true_angular_dist = log_true_angular_dist.apply(
                lambda x: x - log_true_angular_dist_max
            )

            # Convert to a probability distribution, by taking the exponential and normalizing.
            true_angular_dist = log_true_angular_dist.apply(jnp.exp)
            true_angular_dist = true_angular_dist / true_angular_dist.integrate()

            # Check that shapes are correct.
            assert true_angular_dist.grid_values.shape == (
                res_beta,
                res_alpha,
            ), true_angular_dist.grid_values.shape
            assert true_radius_weights.shape == (num_radii,)

            # Mix in the radius weights to get a distribution over all spheres.
            true_dist = true_radius_weights * true_angular_dist[None, :, :]
            # Now, compute the unnormalized predicted distribution over all spheres.
            # Subtract the maximum value for numerical stability.
            log_predicted_dist_max = jnp.max(log_predicted_dist.grid_values)
            log_predicted_dist = log_predicted_dist.apply(
                lambda x: x - log_predicted_dist_max
            )

            # Compute the cross-entropy including a normalizing factor to account for the fact that the predicted distribution is not normalized.
            cross_entropy = -(true_dist * log_predicted_dist).integrate().array.sum()
            normalizing_factor = jnp.log(
                log_predicted_dist.apply(jnp.exp).integrate().array.sum()
            )

            # Compute the self-entropy of the true distribution.
            self_entropy = (
                -(true_dist * true_dist.apply(safe_log)).integrate().array.sum()
            )

            # This should be non-negative, upto numerical precision.
            return cross_entropy + normalizing_factor - self_entropy

        target_positions = graphs.globals.target_positions
        true_radius_weights = jax.vmap(target_position_to_radius_weights)(
            target_positions
        )
        log_true_angular_coeffs = jax.vmap(target_position_to_log_angular_coeffs)(
            target_positions
        )
        log_predicted_dist = position_logits

        assert true_radius_weights.shape == (num_graphs, num_radii)
        assert log_true_angular_coeffs.shape == (
            num_graphs,
            log_true_angular_coeffs.irreps.dim,
        )
        assert log_predicted_dist.grid_values.shape == (
            num_graphs,
            num_radii,
            res_beta,
            res_alpha,
        )

        loss_position = jax.vmap(kl_divergence_on_spheres)(
            true_radius_weights, log_true_angular_coeffs, log_predicted_dist
        )
        assert loss_position.shape == (num_graphs,)
        return loss_position

    def position_loss_with_l2() -> jnp.ndarray:
        """Computes the loss over position probabilities using the L2 loss on the logits."""

        def target_position_to_log_angular_coeffs(
            target_position: jnp.ndarray,
        ) -> e3nn.IrrepsArray:
            """Returns the temperature-scaled angular distribution for a target position."""
            # Compute the true distribution over positions,
            # described by a smooth approximation of a delta function at the target positions.
            norm = jnp.linalg.norm(target_position, axis=-1, keepdims=True)
            target_position_unit_vector = target_position / jnp.where(
                norm == 0, 1, norm
            )
            target_position_unit_vector = e3nn.IrrepsArray(
                "1o", target_position_unit_vector
            )
            return target_position_inverse_temperature * e3nn.s2_dirac(
                target_position_unit_vector,
                lmax=position_coeffs.irreps.lmax,
                p_val=1,
                p_arg=-1,
            )

        def l2_loss_on_spheres(
            true_radius_weights: jnp.ndarray,
            log_true_angular_coeffs: e3nn.IrrepsArray,
            log_predicted_dist_coeffs: e3nn.SphericalSignal,
        ):
            """Computes the L2 loss between the logits of two distributions on the spheres."""
            assert log_true_angular_coeffs.irreps == log_predicted_dist_coeffs.irreps

            log_true_radius_weights = jnp.log(true_radius_weights)[:, None]
            log_true_dist_coeffs_array = jnp.tile(log_true_angular_coeffs.array, (num_radii, 1))
            log_true_dist_coeffs_array = jnp.concatenate([log_true_radius_weights + log_true_dist_coeffs_array[:, :1], log_true_dist_coeffs_array[:, 1:]], axis=1)
            log_true_dist_coeffs = e3nn.IrrepsArray(log_predicted_dist_coeffs.irreps, log_true_dist_coeffs_array)

            assert log_true_dist_coeffs.shape == log_predicted_dist_coeffs.shape, (log_true_dist_coeffs.shape, log_predicted_dist_coeffs.shape)

            norms_of_differences = e3nn.norm(
                log_true_dist_coeffs - log_predicted_dist_coeffs,
                per_irrep=False,
                squared=True,
            ).array.squeeze(-1)

            return jnp.sum(norms_of_differences)

        position_coeffs = preds.globals.position_coeffs
        target_positions = graphs.globals.target_positions
        true_radius_weights = jax.vmap(target_position_to_radius_weights)(
            target_positions
        )
        log_true_angular_coeffs = jax.vmap(target_position_to_log_angular_coeffs)(
            target_positions
        )
        log_predicted_dist_coeffs = position_coeffs

        assert target_positions.shape == (num_graphs, 3)
        assert true_radius_weights.shape == (num_graphs, num_radii)
        assert log_true_angular_coeffs.shape == (
            num_graphs,
            log_true_angular_coeffs.irreps.dim,
        )
        assert log_predicted_dist_coeffs.shape == (
            num_graphs,
            num_radii,
            log_predicted_dist_coeffs.irreps.dim,
        )

        loss_position = jax.vmap(l2_loss_on_spheres)(
            true_radius_weights, log_true_angular_coeffs, log_predicted_dist_coeffs
        )
        assert loss_position.shape == (num_graphs,)
        return loss_position

    def position_loss() -> jnp.ndarray:
        """Computes the loss over position probabilities."""
        if position_loss_type == "kl_divergence":
            return position_loss_with_kl_divergence()
        elif position_loss_type == "l2":
            return position_loss_with_l2()
        else:
            raise ValueError(f"Unsupported position loss type: {position_loss_type}.")

    # If this is the last step in the generation process, we do not have to predict atom type and position.
    loss_focus = focus_loss()
    loss_atom_type = atom_type_loss() * (1 - graphs.globals.stop)
    loss_position = position_loss() * (1 - graphs.globals.stop)

    # Ignore position loss for graphs with less than, or equal to 3 atoms.
    # This is because there are symmetry-based degeneracies in the target distribution for these graphs.
    if ignore_position_loss_for_small_fragments:
        loss_position = jnp.where(graphs.n_node <= 3, 0, loss_position)

    total_loss = loss_focus + loss_atom_type + loss_position
    return total_loss, (
        loss_focus,
        loss_atom_type,
        loss_position,
    )


@functools.partial(jax.profiler.annotate_function, name="get_predictions")
def get_predictions(
    state: train_state.TrainState,
    graphs: datatypes.Fragments,
    rng: Optional[chex.Array],
) -> datatypes.Predictions:
    """Get predictions from the network for input graphs."""
    return state.apply_fn(state.params, rng, graphs)


@functools.partial(jax.jit, static_argnames=["loss_kwargs"])
def train_step(
    state: train_state.TrainState,
    graphs: datatypes.Fragments,
    loss_kwargs: Dict[str, Union[float, int]],
) -> Tuple[train_state.TrainState, metrics.Collection]:
    """Performs one update step over the current batch of graphs."""

    def loss_fn(params: optax.Params, graphs: datatypes.Fragments) -> float:
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

    batch_metrics = Metrics.single_from_model_output(
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
    graphs: datatypes.Fragments,
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
    return Metrics.single_from_model_output(
        total_loss=total_loss,
        focus_loss=focus_loss,
        atom_type_loss=atom_type_loss,
        position_loss=position_loss,
        mask=mask,
    )


def evaluate_model(
    eval_state: train_state.TrainState,
    datasets: Iterator[datatypes.Fragments],
    splits: Iterable[str],
    rng: chex.PRNGKey,
    loss_kwargs: Dict[str, Union[float, int]],
) -> Dict[str, metrics.Collection]:
    """Evaluates the model on metrics over the specified splits."""

    # Loop over each split independently.
    eval_metrics = {}
    for split in splits:
        split_metrics = None

        # Loop over graphs.
        for graphs in datasets[split].as_numpy_iterator():
            graphs = datatypes.Fragments.from_graphstuple(graphs)

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

    # Helper for evaluation.
    def evaluate_model_helper(
        eval_state: train_state.TrainState,
        step: int,
        rng: chex.PRNGKey,
        is_final_eval: bool,
    ) -> Dict[str, metrics.Collection]:
        # Final eval splits are usually larger.
        if is_final_eval:
            splits = ["train_eval_final", "val_eval_final", "test_eval_final"]
        else:
            splits = ["train_eval", "val_eval", "test_eval"]

        # Evaluate the model.
        with report_progress.timed("eval"):
            eval_metrics = evaluate_model(
                eval_state,
                datasets,
                splits,
                rng,
                config.loss_kwargs,
            )

        # Compute and write metrics.
        for split in splits:
            eval_metrics[split] = eval_metrics[split].compute()
            writer.write_scalars(step, add_prefix_to_keys(eval_metrics[split], split))
        writer.flush()

        return eval_metrics

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
    # We set run_in_evaluation_mode as False,
    # because we want to evaluate how the model performs on unseen data.
    eval_net = create_model(config, run_in_evaluation_mode=False)
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
        every_secs=10800,
    )
    hooks = [report_progress, profile]

    # We will record the best model seen during training.
    best_state = None
    step_for_best_state = initial_step
    min_val_loss = jnp.inf

    # Begin training loop.
    logging.info("Starting training.")
    train_metrics = None
    for step in range(initial_step, config.num_train_steps + 1):
        # Log, if required.
        first_or_last_step = step in [initial_step, config.num_train_steps]
        if step % config.log_every_steps == 0 or first_or_last_step:
            if train_metrics is not None:
                writer.write_scalars(
                    step, add_prefix_to_keys(train_metrics.compute(), "train")
                )
            train_metrics = None

        # Evaluate on validation and test splits, if required.
        if step % config.eval_every_steps == 0 or first_or_last_step:
            eval_state = eval_state.replace(params=state.params)

            # Evaluate on validation and test splits.
            rng, eval_rng = jax.random.split(rng)
            eval_metrics = evaluate_model_helper(
                eval_state,
                step,
                eval_rng,
                is_final_eval=False,
            )

            # Note best state seen so far.
            # Best state is defined as the state with the lowest validation loss.
            if eval_metrics["val_eval"]["total_loss"] < min_val_loss:
                min_val_loss = eval_metrics["val_eval"]["total_loss"]
                best_state = state
                step_for_best_state = step

            # Save the current state and best state seen so far.
            with open(os.path.join(checkpoint_dir, f"params_{step}.pkl"), "wb") as f:
                pickle.dump(state.params, f)
            ckpt.save(
                {
                    "state": state,
                    "best_state": best_state,
                    "step_for_best_state": step_for_best_state,
                }
            )

        # Get a batch of graphs.
        try:
            graphs = next(train_iter)
            graphs = datatypes.Fragments.from_graphstuple(graphs)
        except StopIteration:
            logging.info("No more training data. Continuing with final evaluation.")
            break

        # Perform one step of training.
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
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

    # Once training is complete, return the best state and corresponding metrics.
    logging.info(
        "Evaluating best state from step %d at the end of training.",
        step_for_best_state,
    )
    eval_state = eval_state.replace(params=best_state.params)

    # Evaluate on validation and test splits, but at the end of training.
    rng, eval_rng = jax.random.split(rng)
    metrics_for_best_state = evaluate_model_helper(
        eval_state,
        step,
        eval_rng,
        is_final_eval=True,
    )

    # Checkpoint the best state and corresponding metrics seen during training.
    # Save pickled parameters for easy access during evaluation.
    with report_progress.timed("checkpoint"):
        with open(os.path.join(checkpoint_dir, "params_best.pkl"), "wb") as f:
            pickle.dump(best_state.params, f)
        ckpt.save(
            {
                "state": state,
                "best_state": best_state,
                "step_for_best_state": step_for_best_state,
                "metrics_for_best_state": metrics_for_best_state,
            }
        )

    return best_state, metrics_for_best_state
