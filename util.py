import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import s2_grid, _quadrature_weights_soft
import jax
from jax import numpy as jnp
import mace_jax as mace
from mace_jax.tools.gin_model import model
import numpy as np
import time


def get_qm9_data():
    pass


def cross_entropy(labels, predictions):
    """
    labels:
    predictions: predicted
    """
    -1 * np.sum()


def sample_on_s2grid(key, prob_s2, y, alpha, qw):
    """Sample points on the sphere

    Args:
        key (``jnp.ndarray``): random key
        prob_s2 (``jnp.ndarray``): shape ``(y, alpha)``, no need to be normalized
        y (``jnp.ndarray``): shape ``(y,)``
        alpha (``jnp.ndarray``): shape ``(alpha,)``
        qw (``jnp.ndarray``): shape ``(y,)``

    Returns:
        y_index (``jnp.ndarray``): shape ``()``
        alpha_index (``jnp.ndarray``): shape ``()``
    """
    k1, k2 = jax.random.split(key)
    p_ya = prob_s2  # [y, alpha]
    p_y = qw * jnp.sum(p_ya, axis=1)  # [y]
    y_index = jax.random.choice(k1, jnp.arange(len(y)), p=p_y)
    p_a = p_ya[y_index]  # [alpha]
    alpha_index = jax.random.choice(k2, jnp.arange(len(alpha)), p=p_a)
    return y_index, alpha_index


def integral_s2grid(x, quadrature):
    return e3nn.from_s2grid(jnp.exp(x), 0, p_val=1, p_arg=1, quadrature=quadrature).array[0]


class Atom:
    def __init__(self, x, y, z, atom_type):
        self.x = x
        self.y = y
        self.z = z
        self.type = atom_type


class Molecule:
    def __init__(self):
        self.atoms = set([])

    def add(self, atom: Atom):
        self.atoms.add(atom)
