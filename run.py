import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import s2_grid, _quadrature_weights_soft
import haiku as hk
from jax import numpy as jnp
import mace_jax as mace
from mace_jax.tools.gin_datasets import datasets
from model import Model
from util import loss_fn


def run():
    train_loader, valid_loader, test_loader, atomic_energies_dict, r_max = datasets()
