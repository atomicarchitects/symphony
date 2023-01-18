import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import s2_grid, _quadrature_weights_soft
import jax
from jax import numpy as jnp
import mace_jax as mace
from mace_jax.tools.gin_datasets import datasets
from mace_jax.tools.gin_model import model
import numpy as np
import time


def run():
    train_loader, valid_loader, test_loader, atomic_energies_dict, r_max = datasets()
