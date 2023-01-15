import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import s2_grid, _quadrature_weights_soft, to_s2grid
import jax
from jax import numpy as jnp
import numpy as np


def sample_on_s2grid(key, prob_s2, y, alpha, qw):
    """Sample points on the sphere

    Args:
        key (jnp.ndarray): random key
        prob_s2 (jnp.ndarray): shape ``(y, alpha)``, no need to be normalized
        y (jnp.ndarray): shape ``(y,)``
        alpha (jnp.ndarray): shape ``(alpha,)``
        qw (jnp.ndarray): shape ``(y,)``

    Returns:
        y_index (jnp.ndarray): shape ``()``
        alpha_index (jnp.ndarray): shape ``()``
    """
    k1, k2 = jax.random.split(key)
    p_ya = prob_s2  # [y, alpha]
    p_y = qw * jnp.sum(p_ya, axis=1)  # [y]
    y_index = jax.random.choice(k1, jnp.arange(len(y)), p=p_y)
    p_a = p_ya[y_index]  # [alpha]
    alpha_index = jax.random.choice(k2, jnp.arange(len(alpha)), p=p_a)
    return y_index, alpha_index
