import e3nn_jax as e3nn
from e3nn_jax import to_s2grid, _sh_alpha, _sh_beta, _expand_matrix, _rollout_sh
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from model import sample, DISTANCES
import numpy as np
import time


def loss_fn(key, weights, mace_input, y, lmax, res_beta, res_alpha, quadrature, gamma=30):
    output = sample(key, weights, mace_input, res_beta, res_alpha, quadrature)
    
    ## focus loss
    loss_focus = -1 * jnp.log(output["focus_logits"][y["focus"]]) + logsumexp(output["focus_logits"])
    if output["stop"]:
        return loss_focus

    ## atom type loss
    loss_type = -1 * jnp.log(output["atom_type_logits"][y["atom_type"]]) + logsumexp(output["atom_type_logits"])

    ## position loss
    position_signal = to_s2grid(output["position_coeffs"], res_beta, res_alpha, quadrature=quadrature)
    position_signal = position_signal.apply(jnp.exp)
    prob_radius = position_signal.integrate()

    # getting f(r*, rhat*) [aka, our model's output for the true location]
    sh_a = _sh_alpha(lmax, y["alpha"])  # [1, 2 * lmax + 1]
    sh_y = _sh_beta(lmax, y["y"])  # [1, (lmax + 1) * (lmax + 2) // 2 + 1]
    sh_y = _rollout_sh(sh_y, lmax)
    # exact dimensions will be figured out later
    true_eval_pos = jnp.sum(sh_y * sh_a * output["position_coeffs"])

    # get radius weights for the "true" distribution
    radius_weights = jax.vmap(
        lambda dist_bucket: jnp.exp(-1 * (y["radius"] - dist_bucket)**2 / gamma),
        DISTANCES
    )
    radius_weights = radius_weights / jnp.sum(radius_weights)
    loss_pos = -true_eval_pos + jnp.log(jnp.sum(prob_radius * radius_weights))

    ## return total loss
    return loss_focus + loss_type + loss_pos


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


def integral_s2grid(x, quadrature, p_val=1, p_arg=-1):
    return e3nn.from_s2grid(x, 0, p_val, p_arg, quadrature=quadrature)


class Atom:
    def __init__(self, x, y, z, atom_type):
        self.x = x
        self.y = y
        self.z = z
        self.type = atom_type
