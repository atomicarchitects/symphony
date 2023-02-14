from collections import namedtuple
import e3nn_jax as e3nn
from e3nn_jax import Irreps
import haiku as hk
import jax
from jax import numpy as jnp
import jraph
from mace_jax.data import GraphNodes, GraphEdges, GraphGlobals
from mace_jax.modules import GeneralMACE
import optax
import tqdm

from datatypes import WeightTuple, MaceInput, ModelOutput
from util import _loss, TYPES, DISTANCES


@hk.without_apply_rng
@hk.transform
def mace_fn(mace_input):
    return GeneralMACE(
        output_irreps="128x0e + 32x1o + 32x2e + 32x3o + 32x4e + 32x5o",
        r_max=5,
        num_interactions=2,
        hidden_irreps="128x0e + 128x1o + 128x2e",
        readout_mlp_irreps="128x0e + 128x1o + 128x2e",
        avg_num_neighbors=3,  # idk
        num_species=5,
        radial_basis=lambda x, x_max: e3nn.bessel(x, 8, x_max),
        radial_envelope=e3nn.soft_envelope,
        max_ell=3,
    )(
        mace_input.vectors,
        mace_input.atom_types,
        mace_input.senders,
        mace_input.receivers,
    )


mace_apply = jax.jit(mace_fn.apply)


@hk.without_apply_rng
@hk.transform
def focus_fn(x):
    # models to map each atom to P(atom = focus)
    # inputs: feature vectors for 1 atom
    # outputs: P(atom = focus) for each atom (so 2 numbers - p and 1-p?), also P(stop)
    return e3nn.haiku.MultiLayerPerceptron(
        list_neurons=[128, 128, 128, 64, 16, 1],
        act=jax.nn.relu,  # idk I just chose one
    )(x)


@hk.without_apply_rng
@hk.transform
def atom_type_fn(x):
    # choosing element type for the next atom
    # inputs: atom feature vectors
    # outputs: P(H), P(C), P(N), P(O), P(F)
    return e3nn.haiku.MultiLayerPerceptron(
        list_neurons=[128, 128, 128, 128, 128, 128, 5], act=jax.nn.softplus
    )(x)


@hk.without_apply_rng
@hk.transform
def position_fn(x, z):
    """
    Args:
        x (`e3nn.IrrepsArray`): features for focus atom
        z (int): atom type
    """
    # get atom type embedding
    z = hk.Embed(5, 128)(z)  # (128)
    # radial/angular distribution
    return e3nn.haiku.Linear(Irreps("64x0e+64x1o"))(
        x * z
    )  # this architecture is temporary


def model_run(w: WeightTuple, mace_input: MaceInput):
    """Runs one step of the model"""

    # get feature representations
    features = mace_apply(w.mace, mace_input)  # (atoms, irreps)

    # get focus
    focus_logits = jnp.concatenate(
        [focus_fn.apply(w.focus, features), jnp.array([0])]  # (atoms)
    )
    focus_probs = jax.nn.softmax(focus_logits)
    key, new_key = jax.random.split(key)
    focus_pred = jax.random.choice(new_key, jnp.arange(len(focus_probs)), p=focus_probs)
    focus_true = jnp.concatenate([jnp.array([0]), jnp.cumsum(mace_input.n_node[:-1])])

    # get atom type
    atom_type_logits = atom_type_fn.apply(w.atom_type, features[focus_true])
    atom_type_dist = jax.nn.softmax(atom_type_logits)
    key, new_key = jax.random.split(key)
    atom_type = jax.random.choice(new_key, jnp.arange(5), p=atom_type_dist)

    # get position distribution
    position_coeffs = position_fn.apply(
        w.position, features[focus_true], atom_type
    )  # IrrepsArray (300, irreps)

    return ModelOutput(
        focus_pred == len(focus_probs) - 1,  # stop (global)
        focus_logits,  # focus (node)
        atom_type_logits,  # atom type (global)
        position_coeffs,  # postiion (global)
    )


def train(data_loader, res_beta, res_alpha, quadrature, gamma=30, learning_rate=1e-4):

    w_mace = mace_fn.init(jax.random.PRNGKey(0), mace_input)
    w_focus = focus_fn.init(jax.random.PRNGKey(0), jnp.zeros((2, 128)))
    w_type = atom_type_fn.init(jax.random.PRNGKey(0), jnp.zeros((2, 128)))
    w_position = position_fn.init(
        jax.random.PRNGKey(0), jnp.zeros((2, 128)), jnp.zeros((2,), dtype=jnp.int32)
    )

    weights = WeightTuple(w_mace, w_focus, w_type, w_position)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)

    datapoints_bar = tqdm.tqdm(
        data_loader, desc="Training", total=data_loader.approx_length()
    )
    for graph in datapoints_bar:
        vectors = (
            graph.nodes.positions[graph.receivers]
            - graph.nodes.positions[graph.senders]
        )
        atom_types = graph.nodes.species
        mace_input = MaceInput(vectors, atom_types, graph.senders, graph.receivers)
        weights, opt_state = _train(
            graph, weights, opt_state, optimizer, res_beta, res_alpha, quadrature, gamma
        )


@jax.jit
def _train(graph, weights, state, optim, res_beta, res_alpha, quadrature, gamma=30):
    loss, grad = jax.value_and_grad(loss_fn)(
        graph, weights, res_beta, res_alpha, quadrature, gamma
    )
    updates, state = optim.update(grad, state, w)
    w = optax.apply_updates(w, updates)
    return w, state


def loss_fn(graph, weights, res_beta, res_alpha, quadrature, gamma=30):
    output = model_run(weights, graph)

    return _loss(output, graph, res_beta, res_alpha, quadrature, gamma)


def evaluate(weights, data_loader, res_beta, res_alpha, quadrature):
    datapoints_bar = tqdm.tqdm(
        data_loader, desc="Evaluating", total=data_loader.approx_length()
    )
    for data in datapoints_bar:
        # what format is data in? look at jraph.ipynb
        output = model_run(jax.random.PRNGKey(0), weights, mace_input)
        loss = _loss(output, graph, res_beta, res_alpha, quadrature, gamma)
