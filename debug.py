import functools
import jax
import jax.numpy as jnp
import jraph
import optax
import sys
import ml_collections
import e3nn_jax as e3nn
import plotly.graph_objects as go
from flax.training import train_state
import tqdm

from symphony.data import input_pipeline_tf, input_pipeline
from symphony import models, train, loss
from configs.platonic_solids import e3schnet_and_nequip as config_platonic

from jax import config
config.update("jax_debug_nans", True)
config_platonic = config_platonic.get_config()
config_platonic = ml_collections.FrozenConfigDict(config_platonic)

# Load the dataset.
datasets = input_pipeline_tf.get_datasets(jax.random.PRNGKey(0), config_platonic)
for step, graphs in enumerate(datasets["train"].as_numpy_iterator()):
    graphs = jax.tree_map(jnp.asarray, graphs)
    for graph in jraph.unbatch(graphs):
        if jnp.sum(graph.globals.target_position_mask) > 1:
            fragment = graph
            break
    break


num_radii = 20
radial_bins = jnp.linspace(0.5, 1.5, num_radii)
loss_kwargs = config_platonic.loss_kwargs
num_steps = 10000
report_every = num_steps // 50
rng = jax.random.PRNGKey(0)


@jax.jit
def train_step(
    graphs,
    state,
    rng,
    noise_std: float,
):
    """Performs one update step over the current batch of graphs."""

    loss_rng, rng = jax.random.split(rng)
    def loss_fn(params, graphs,) -> float:
        curr_state = state.replace(params=params)
        preds = train.get_predictions(curr_state, graphs, rng=loss_rng)
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

    # # Add noise to positions, if required.
    # if add_noise_to_positions:
    noise_rng, rng = jax.random.split(rng)
    position_noise = (
        jax.random.normal(noise_rng, graphs.nodes.positions.shape) * noise_std
    )
    # else:
    # position_noise = jnp.zeros_like(graphs.nodes.positions)

    noisy_positions = graphs.nodes.positions + position_noise
    graphs = graphs._replace(nodes=graphs.nodes._replace(positions=noisy_positions))

    # Compute gradients.
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (
        _,
        (total_loss, focus_and_atom_type_loss, position_loss, mask),
    ), grads = grad_fn(state.params, graphs)

    # Average gradients across devices.
    # grads = jax.lax.pmean(grads, axis_name="device")
    state = state.apply_gradients(grads=grads)

    return state, total_loss, focus_and_atom_type_loss, position_loss


for learning_rate in [1e-2]:
# for learning_rate in [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]:
    
    init_rng, rng = jax.random.split(rng)
    model = models.create_model(config_platonic, run_in_evaluation_mode=False)
    params = model.init(init_rng, graphs)
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    for step in tqdm.tqdm(range(num_steps)):
        step_rng, rng = jax.random.split(rng)
        state, total_loss, focus_and_atom_type_loss, position_loss = train_step(
            graphs,
            state,
            step_rng,
            0.0,
        )
        if step % report_every == 0 or step == num_steps - 1:
            print(f"step={step}: mean position loss={position_loss.mean()}")
