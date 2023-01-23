import e3nn_jax as e3nn
from e3nn_jax import from_s2grid
import jax
from jax import numpy as jnp
import numpy as np
import time


def loss_fn(type_a, type_dist, rad_a, rad_p, ang_a, angular_dist_grid, lmax, quadrature):
    """
    Args:
        type_a (): actual type
        type_p (``jnp.ndarray``): predicted type distribution
        rad_a (float): index of actual distance from focus
        rad_p (``e3nn.IrrepsArray``): predicted radial distribution
        ang_a (float): index of actual angular position
        angular_dist_grid (``e3nn.IrrepsArray``): predicted angular distribution
        res_beta (int): number of points on the sphere in the :math:`\theta` direction
        res_alpha (int): number of points on the sphere in the :math:`\phi` direction
        quadrature (str): "soft" or "gausslegendre"
    """
    # type loss
    loss_type = type_a * jnp.sum(jnp.log(type_dist))

    # radial loss
    radial_dist = from_s2grid()

    # angular loss
    angular_dist = from_s2grid(angular_dist_grid, lmax, quadrature=quadrature, p_val=p_val, p_arg=p_arg)
    # p_val and p_arg should match the original output of the NN
    angular_max_prob = jnp.max(angular_dist_grid)
    angular_log = jnp.log(integral_s2grid(jnp.exp(angular_dist_grid - angular_max_prob), quadrature))
    loss_ang = s(x) - angular_max_prob - angular_log

    return -1 * (loss_type + loss_rad + loss_ang)


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
