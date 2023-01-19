import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import s2_grid, _quadrature_weights_soft
import jax
from jax import numpy as jnp
import logging
import mace_jax as mace
from mace_jax.tools.gin_model import model
import numpy as np
import tqdm

from util import Atom, cross_entropy, sample_on_s2grid


TYPE_MAP = ["STOP", "H", "C", "N", "O", "F"]


## this needs to become a haiku module
class Model:
    def __init__(self, feature_model_args, distance_model_args, seed):
        # feature extraction models
        # inputs: types + positions of the atoms constructed thus far
        # outputs: equivariant features used by the model

        # feature_params: weights used in the NN
        self.feature_model, self.feature_params, num_message_passing = model(
            **feature_model_args, initialize_seed=seed
        )

        # models to map each atom to P(atom = focus)
        # inputs: atom feature vectors
        # outputs: P(atom = focus) for each atom (so 2 numbers - p and 1-p?)
        self.focus_model = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[
                128,
                128,
                128,
                64,
                16,
                2,
            ],  # idk, but I know I want 2 at the end
            act=jax.nn.relu,  # idk I just chose one
        )
        self.focus_model.init(
            jax.random.PRNGKey(seed), jnp.ones()
        )  # make this key choice random as well

        # choosing element type for the next atom
        # inputs: atom feature vectors
        # outputs: H, C, N, O, F, STOP
        self.type_model = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[128, 128, 128, 128, 128, 128, 6], act=jax.nn.softplus
        )
        self.focus_model.init(
            jax.random.PRNGKey(seed), jnp.ones()
        )  # make this key choice random as well

        # radial/angular distribution
        # inputs: focus atom's features, focus atom type
        # outputs: distributions for radial and angular distributions
        self.position_model, self.position_params, num_message_passing = e3nn.haiku.

    def __call__(self, input_data):
        """Single step forward (one atom generated), not the entire generation process"""
        # get feature representations
        features_out = self.feature_model(input_data)

        # get focus
        focus_probs = [0, 1]
        focus = -1
        for atom in range(len(features_out)):
            probs = jax.nn.softmax(self.focus_model(features_out[atom]))
            if probs[0] > focus_probs:  # whichever of them is the "P(focus)" one
                focus_probs = probs
                focus = atom
        assert len(probs) == 2 and probs[0] + probs[1] == 1

        # get atom type
        type_inputs = np.concatenate([np.asarray(features_out[atom])])
        type_outputs = jax.nn.softmax(self.type_model(type_inputs))
        type = np.argmax(type_outputs)
        if type == 0:  # stop token; arbitrary choice of index
            return "STOP", []

        # get distance distribution
        distance_input = np.concatenate(
            [np.asarray(features_out[atom]), np.asarray([type]), np.asarray([focus])]
        )
        distance_dist = self.distance_model(distance_input)

        return TYPE_MAP[type], distance_dist


def train(model: Model, loss):
    def update_fn():
        pass

    # run through the whole generate process and compare results?


def evaluate(model: Model, data_loader):
    p_bar = tqdm.tqdm(data_loader, desc="Evaluating", total=data_loader.approx_length())
    for ref_graph in p_bar:
        output = model(ref_graph)
        # how does this work when we aren't dealing with forces/energy/stresses?
        


def generate(model, input_data, res_beta, res_alpha, quadrature):
    """Entire generation process; creates a molecule atom-by-atom until a stop token is reached"""
    output_molecule = []
    curr_atom_type = "X"
    while curr_atom_type != "STOP":
        ## generate position distribution
        curr_atom_type, distance_dist = model(input_data)
        if curr_atom_type == "STOP":
            break

        ## sample position distribution
        # this needs to be changed if we're predicting radial and angular distributions together
        f = e3nn.to_s2grid(distance_dist, (res_beta, res_alpha), quadrature=quadrature)
        Z = e3nn.from_s2grid(
            jnp.exp(f), 0, p_val=1, p_arg=1, quadrature=quadrature
        ).array[
            0
        ]  # integral of exp(f) on S2
        p = jnp.exp(f) / Z

        y_orig, alpha_orig = s2_grid(res_beta, res_alpha, quadrature=quadrature)
        y, alpha = jnp.meshgrid(y_orig, alpha_orig, indexing="ij")
        x = jnp.cos(alpha) * jnp.sqrt(1 - y**2)
        z = jnp.sin(alpha) * jnp.sqrt(1 - y**2)

        if quadrature == "soft":
            qw = _quadrature_weights_soft(res_beta // 2) * res_beta**2  # [b]
        elif quadrature == "gausslegendre":
            _, qw = np.polynomial.legendre.leggauss(res_beta)
            qw /= 2
        else:
            Exception('quadrature should be either "soft" or "gausslegendre"')

        sampled_y_i, sampled_alpha_i = sample_on_s2grid(
            jax.random.PRNGKey(0), p, y_orig, alpha_orig, qw
        )
        sampled_y = y_orig[sampled_y_i]
        sampled_alpha = alpha_orig[sampled_alpha_i]
        sampled_x = jnp.cos(sampled_alpha) * jnp.sqrt(1 - sampled_y**2)
        sampled_z = jnp.sin(sampled_alpha) * jnp.sqrt(1 - sampled_y**2)

        # also need to keep track of which atoms are connected!
        output_molecule.append(Atom(sampled_x, sampled_y, sampled_z, curr_atom_type))

    return output_molecule
