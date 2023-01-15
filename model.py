import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import s2_grid, _quadrature_weights_soft, to_s2grid
import jax
from jax import numpy as jnp
import numpy as np
from sampling import sample_s2grid_angular


class Model:
    pass


def train():
    pass
    # run through the whole generate process and compare results?


def generate():
    pass
    ### while atom type != "stop":

    ## pick a focus:
    #   input: array of all atoms
    #   output: likelihood of each atom being the focus

    ## predict atom element type

    ## generate radial distribution; sample a distance

    ## generate angular distribution

    ## sample from the angular distribution
