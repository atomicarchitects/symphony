from collections import namedtuple
import e3nn_jax as e3nn
from e3nn_jax import Irreps, IrrepsArray, s2_grid, _quadrature_weights_soft
import haiku as hk
import jax
from jax import numpy as jnp
import jraph
import mace_jax as mace
import numpy as np
import optax
import tqdm

from util import Atom, integral_s2grid, loss_fn, sample_on_s2grid


TYPES = ["H", "C", "N", "O", "F", "STOP"]


@hk.transform
def mace_fn(x):
    feature_model_args = {
        'r_max': 15,
        # train_graphs?
    }
    # return model_.apply, params, num_interactions
    return mace.tools.gin_model.model(**feature_model_args)


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


def sample(key, w: namedtuple, g: jraph.GraphTuple):
    """Runs one step of the model"""

    # get feature representations
    features = mace_fn.apply(w.mace, g)  # (atoms, irreps)

    # get focus
    focus_logits = jnp.concatenate(
        [focus_fn.apply(w.focus, features), jnp.array([0])]  # (atoms)
    )
    focus_probs = jax.nn.softmax(focus_logits)
    key, new_key = jax.random.split(key)
    focus = jax.random.choice(new_key, jnp.arange(len(focus_probs)), p=focus_probs)

    # get atom type
    atom_type_dist = jax.nn.softmax(atom_type_fn.apply(w.atom_type, features[focus]))
    key, new_key = jax.random.split(key)
    atom_type = jax.random.choice(new_key, jnp.arange(5), p=atom_type_dist)

    # get position distribution
    position_coeffs = position_fn.apply(
        w.position, features[focus], atom_type
    )  # IrrepsArray (300, irreps)

    # get radial distribution
    position_signal = e3nn.to_s2grid(position_coeffs, res_beta, res_alpha, quadrature=quadrature)  # (300, beta, alpha)
    position_signal = position_signal.apply(jnp.exp)
    prob_radius = position_signal.integrate()  # (300)

    # sample a distance
    key, new_key = jax.random.split(key)
    r_ndx = jax.random.choice(new_key, distance_bins, prob_radius)

    # get angular distribution
    angular_coeffs = position_coeffs[r_ndx]  # (irreps)
    angular_signal = e3nn.to_s2grid(angular_coeffs, res_beta, res_alpha, quadrature=quadrature)  # (irreps, beta, alpha)
    angular_signal = angular_signal.apply(jnp.exp)
    prob_angle = angular_signal.integrate()  # (irreps)

    # sample angular distribution
    y, alpha = s2_grid(res_beta, res_alpha, quadrature=quadrature)

    if quadrature == "soft":
        qw = _quadrature_weights_soft(res_beta // 2) * res_beta**2  # (b)
    elif quadrature == "gausslegendre":
        _, qw = np.polynomial.legendre.leggauss(res_beta)
        qw /= 2
    else:
        Exception('quadrature should be either "soft" or "gausslegendre"')

    key, new_key = jax.random.split(key)
    sampled_y_i, sampled_alpha_i = sample_on_s2grid(new_key, prob_angle, y, alpha, qw)
    sampled_y = y[sampled_y_i]
    sampled_alpha = alpha[sampled_alpha_i]

    return {
        "stop": focus == len(focus_probs) - 1,
        "atom_type_dist": atom_type_dist,
        "radial_dist": prob_radius,
        "angular_dist": prob_angle,
        "type_pred": atom_type,
        "r_ndx_pred": r_ndx,
        "y_pred": sampled_y,
        "alpha_pred": sampled_alpha,
    }


weight_tuple = namedtuple("WeightTuple", ["mace", "focus", "atom_type", "position"])


def train(data_loader, learning_rate=1e-4):
    w_mace = mace_fn.init(jax.random.PRNGKey(0), jnp.zeros((2, 32)))
    w_focus = focus_fn.init(jax.random.PRNGKey(0), jnp.zeros((2, 128)))
    w_type = atom_type_fn.init(jax.random.PRNGKey(0), jnp.zeros((2, 128)))
    w_position = position_fn.init(
        jax.random.PRNGKey(0), jnp.zeros((2, 128)), jnp.zeros((2,), dtype=jnp.int32)
    )

    weights = weight_tuple(w_mace, w_focus, w_type, w_position)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)

    datapoints_bar = tqdm.tqdm(
        data_loader, desc="Training", total=data_loader.approx_length()
    )
    for data in datapoints_bar:
        x = 
        y = 
        weights, opt_state = _train(weights, x, y, opt_state)


@jax.jit
def _train(w, x, y, state, optim):
    loss, grad = jax.value_and_grad(loss_fn)(w, x, y)
    updates, state = optim.update(grad, state, w)
    w = optax.apply_updates(w, updates)
    return w, state


def evaluate(model, data_loader, res_beta, res_alpha, quadrature):
    datapoints_bar = tqdm.tqdm(
        data_loader, desc="Evaluating", total=data_loader.approx_length()
    )
    for data in datapoints_bar:
        output = model(data, res_beta, res_alpha, quadrature)


def generate(model, input_data, res_beta, res_alpha, quadrature):
    """Entire generation process; creates a molecule atom-by-atom until a stop token is reached"""
    output_molecule = set()
    while True:
        ## get atom type, generate position distribution
        output = model(input_data, res_beta, res_alpha, quadrature)
        if output["stop"]:
            break

        sampled_x = jnp.cos(output["alpha_pred"]) * jnp.sqrt(1 - output["y_pred"] ** 2)
        sampled_z = jnp.sin(output["alpha_pred"]) * jnp.sqrt(1 - output["y_pred"] ** 2)

        output_molecule.add(
            Atom(
                sampled_x * output["radius"],
                output["y_pred"] * output["radius"],
                sampled_z * output["radius"],
                output["type_pred"],
            )
        )

    return output_molecule
