import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import s2_grid, _quadrature_weights_soft
import haiku as hk
import jax
from jax import numpy as jnp
import logging
import mace_jax as mace
from mace_jax.tools.gin_model import model
import numpy as np
import tqdm

from util import Atom, Molecule, cross_entropy, integral_s2grid, sample_on_s2grid


def get_qm9_data():
    # this should return data in Molecule form
    pass


def preprocess():
    # this should return data in a form usable by the MACE model
    pass
