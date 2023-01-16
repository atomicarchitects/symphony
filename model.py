import e3nn
import torch
import mace
import numpy as np
import jax
from jax import numpy as jnp
import typing


class Model:
    def __init__(self, quadrature):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # feature extraction models
        # inputs: types + positions of the atoms constructed thus far
        # outputs: equivariant features used by the model
        feature_model_config = dict(
            r_max=args.r_max,
            num_bessel=args.num_radial_basis,
            num_polynomial_cutoff=args.num_cutoff_basis,
            max_ell=args.max_ell,
            interaction_cls=modules.interaction_classes[args.interaction],
            num_interactions=args.num_interactions,
            num_elements=len(z_table),
            hidden_irreps=o3.Irreps(args.hidden_irreps),
            atomic_energies=atomic_energies,
            avg_num_neighbors=args.avg_num_neighbors,
            atomic_numbers=z_table.zs,
            correlation=args.correlation,
            gate=modules.gate_dict[args.gate],
            interaction_cls_first=modules.interaction_classes[args.interaction_first],
            MLP_irreps=o3.Irreps(args.MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=mean,
        )
        self.feature_model = mace.modules.ScaleShfitMACE(**feature_model_config)
        self.feature_model.to(device)

        # models to map each atom to P(atom = focus)
        # inputs: atom feature vectors
        # outputs: P(atom = focus) for each atom (so 2 numbers - p and 1-p?)
        self.focus_model = e3nn.flax.MultiLayerPerceptron(
            list_neurons=[128, 128, 128, 64, 16, 2],  # idk, but I know I want 2 at the end
            act=jax.nn.relu  # idk I just chose one
        )  # should get softmax'ed or something at the end?
        self.focus_model.init(jax.random.PRNGKey(0), jnp.ones())  # make this key choice random as well

        # choosing element type for the next atom
        # inputs: atom feature vectors
        # outputs: H, C, N, O, F, STOP
        self.type_model = e3nn.flax.MultiLayerPerceptron(
            list_neurons=[128, 128, 128, 128, 128, 128, 6],
            act=jax.nn.softplus
        )  # should get softmax'ed at the end
        self.focus_model.init(jax.random.PRNGKey(0), jnp.ones())  # make this key choice random as well

        # distance distribution
        # inputs: focus atom's features, focus atom type
        # outputs: distance distribution
        self.distance_model = mace.modules.ScaleShiftMACE(**distance_model_config)
        self.distance_model.to(device)


    def forward(self, input_data):
        # get feature representations
        features_out = self.feature_model(input_data)

        # get focus
        focus_probs = [0, 1]
        focus = -1
        for atom in range(len(features_out)):
            probs = self.focus_model(features_out[atom])
            if probs[0] > focus_probs:  # whichever of them is the "P(focus)" one
                focus_probs = probs
                focus = atom
        assert(len(probs) == 2 and probs[0] + probs[1] == 1)

        # get atom type
        type_inputs = np.concatenate([np.asarray(features_out[atom])])
        type_outputs = self.type_model(type_inputs)
        type_outputs = np.exp(type_outputs) / np.sum(np.exp(type_outputs))
        type = np.argmax(type_outputs)

        # get distance distribution
        distance_input = np.concatenate([np.asarray(features_out[atom]), np.asarray([type])])
        distance_dist = self.distance_model(distance_input)


def train():
    pass
    # run through the whole generate process and compare results?


def evaluate():
    pass


def generate():
    output_molecule = []
    curr_atom_type = "X"
    while curr_atom_type != "STOP":
        ## pick a focus:
        #   input: array of all atoms
        #   output: likelihood of each atom being the focus

        ## predict atom element type

        ## generate position distribution

        ## sample position distribution
        f = e3nn.to_s2grid(positionDist, (res_beta, res_alpha), quadrature=quadrature)
        Z = e3nn.from_s2grid(jnp.exp(f), 0, p_val=1, p_arg=1, quadrature=quadrature).array[0]  # integral of exp(f) on S2
        p = jnp.exp(f) / Z
