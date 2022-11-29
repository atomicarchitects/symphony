import e3nn_jax as e3nn
from e3nn_jax._src.s2grid import s2_grid, _quadrature_weights_soft, to_s2grid
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import typing


def sample_s2grid_angular(proba, res_beta, res_alpha, *, quadrature: str, key: jax.random.PRNGKey):
    r"""
    Take samples from a signal on the S2 grid.
    Args:
        proba (`jax.numpy.ndarray`): signal on the sphere of shape ``(res_beta, res_alpha)``
        num_samples (int): the number of samples to take from proba
        quadrature (str): "soft" or "gausslegendre"
        key (jax.random.PRNGKey)
    Returns:
        sampled_z (float): sampled z taken from proba
        sampled_alpha (float): sampled alpha taken from proba
    """
    zs, alphas = s2_grid(res_beta, res_alpha, quadrature=quadrature)
    if quadrature == "soft":
        qw = _quadrature_weights_soft(res_beta // 2) * res_beta**2  # [b]
    elif quadrature == "gausslegendre":
        _, qw = np.polynomial.legendre.leggauss(res_beta)
        qw /= 2
    p_z = jnp.sum(proba, axis=-1) * qw
    z_i = jax.random.choice(key, jnp.arange(proba.shape[0]), p=p_z)
    sampled_z = zs[z_i]
    sampled_alpha = jax.random.choice(key, alphas, p=proba[z_i])
    
    return sampled_z, sampled_alpha
