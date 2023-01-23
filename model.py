import e3nn_jax as e3nn
from e3nn_jax import Irreps, IrrepsArray, s2_grid, _quadrature_weights_soft
import haiku as hk
import jax
from jax import numpy as jnp
from mace_jax.tools.gin_model import model
import numpy as np
import optax
import tqdm

from util import Atom, Molecule, integral_s2grid, loss_fn, sample_on_s2grid


## this needs to become a haiku module
# also need to include atom type embeddings
class Model(hk.Module):
    def __init__(self, feature_model_args, distance_model_args, seed=0, dist_bins=300, dist_step=0.05):
        self.dist_bins = dist_bins
        self.dist_step = dist_step

        # feature extraction models
        # inputs: types + positions of the atoms constructed thus far
        # outputs: equivariant features used by the model
        self.feature_model, self.feature_params, num_message_passing = model(
            **feature_model_args, initialize_seed=seed
        )

        # models to map each atom to P(atom = focus)
        # inputs: atom feature vectors
        # outputs: P(atom = focus) for each atom (so 2 numbers - p and 1-p?), also P(stop)
        self.focus_model = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[128, 128, 128, 64, 16, 2],  # idk, but I know I want 2 at the end
            act=jax.nn.relu,  # idk I just chose one
        )

        # choosing element type for the next atom
        # inputs: atom feature vectors
        # outputs: P(H), P(C), P(N), P(O), P(F)
        self.type_model = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[128, 128, 128, 128, 128, 128, 6], act=jax.nn.softplus
        )

        # radial/angular distribution
        # inputs: focus atom's features, focus atom type
        # outputs: distributions for radial and angular distributions (dims 300 x [# of sh terms])
        # will need to get distance distribution by marginalizing over the angular distribution
        self.position_model, self.position_params, num_message_passing = e3nn.haiku.Linear(
            Irreps('64x0e+64x1o')
        )  # this architecture is temporary

        # atom type embedding: IrrepsArray
        # shape (5, t)
        self.type_embedding = jnp.ones((5, 128))

    def __call__(self, x):
        """Single step forward (one atom generated), not the entire generation process"""
        # get feature representations
        features_out = self.feature_model(x)

        # get focus
        focus_probs = jax.nn.softmax(self.focus_model(features_out))
        focus = jnp.argmax(jnp.concatenate([focus_probs, 0]))  # the 0 is a stand-in for "stop"
        # if "stop" has the highest probability, stop
        if focus == 5:
            return "STOP", []

        # get atom type
        # these concatenates could become element-wise multiplications
        type_inputs = jnp.concatenate([jnp.asarray(features_out[focus])])
        type_outputs = jax.nn.softmax(self.type_model(type_inputs))  # jnp.ndarray
        type = jnp.argmax(type_outputs)

        # get position distribution
        position_input = jnp.concatenate(
            [jnp.asarray(features_out[focus]), self.type_embedding[type]]
        )
        position_distributions = self.distance_model(position_input)  # IrrepsArray

        types = ['H', 'C', 'N', 'O', 'F']
        return types[type], position_distributions


def train(model_w_loss, data_loader, learning_rate, key, N):
    '''
    Args:
        model_w_loss: transformed function that runs a model and returns the corresponding loss
        data_loader
        learning_rate
        key (jax.PRNGKey)
        N (int)
    '''
    optimizer = optax.adam(learning_rate)
    loss_and_grad_fn = jax.value_and_grad(model_w_loss.apply)
    data = next(data_loader)
    params = model_w_loss.init(key, data)
    opt_state = optimizer.init(params)

    for i in range(N):
        data = next(data_loader)
        key, key_new = jax.random.split(key)
        loss, gradients = loss_and_grad_fn(params, key_new, data)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        params = optax.apply_updates(params, updates)
        if i % 10 == 0:
            print(f'Step {i}: loss {loss}')

    return params


def evaluate(model: Model, data_loader):
    p_bar = tqdm.tqdm(data_loader, desc="Evaluating", total=data_loader.approx_length())
    for ref_graph in p_bar:
        output = model(ref_graph)
        # how does this work when we aren't dealing with forces/energy/stresses?
        


def generate(model, input_data, res_beta, res_alpha, quadrature, seed=0):
    """Entire generation process; creates a molecule atom-by-atom until a stop token is reached"""
    output_molecule = Molecule()
    curr_atom_type = "X"
    while curr_atom_type != "STOP":
        ## get atom type, generate position distribution
        curr_atom_type, position_distributions = model(input_data)
        if curr_atom_type == "STOP":
            break

        ## get radial distribution, sample a distance
        radial_dist = e3nn.sum(position_distributions, axis=-1)
        lmax_r = radial_dist.irreps.ls[-1]
        f = e3nn.to_s2grid(radial_dist, (res_beta, res_alpha), quadrature=quadrature)
        Z = integral_s2grid(f, quadrature)
        radial_dist = jnp.exp(f) / Z
        r_ndx = jax.random.choice(model.key, model.dist_bins, radial_dist)

        ## get angular distribution
        angular_dist = position_distributions[r_ndx]
        lmax_a = angular_dist.irreps.ls[-1]
        f = e3nn.to_s2grid(angular_dist, (res_beta, res_alpha), quadrature=quadrature)
        Z = integral_s2grid(f, quadrature)
        angular_dist = jnp.exp(f) / Z

        ## sample angular distribution
        y, alpha = s2_grid(res_beta, res_alpha, quadrature=quadrature)
    
        if quadrature == "soft":
            qw = _quadrature_weights_soft(res_beta // 2) * res_beta**2  # [b]
        elif quadrature == "gausslegendre":
            _, qw = np.polynomial.legendre.leggauss(res_beta)
            qw /= 2
        else:
            Exception('quadrature should be either "soft" or "gausslegendre"')

        sampled_y_i, sampled_alpha_i = sample_on_s2grid(
            jax.random.PRNGKey(seed), angular_dist, y, alpha, qw
        )
        sampled_y = y[sampled_y_i]
        sampled_alpha = alpha[sampled_alpha_i]
        sampled_x = jnp.cos(sampled_alpha) * jnp.sqrt(1 - sampled_y**2)
        sampled_z = jnp.sin(sampled_alpha) * jnp.sqrt(1 - sampled_y**2)

        # also need to keep track of which atoms are connected!
        output_molecule.add(Atom(sampled_x, sampled_y, sampled_z, curr_atom_type))

    return output_molecule
