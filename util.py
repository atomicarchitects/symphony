import e3nn_jax as e3nn
from e3nn_jax import to_s2grid
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from model import sample
import numpy as np
import time


def loss_fn(key, weights, mace_input, y, res_beta, res_alpha, quadrature):
    output = sample(key, weights, mace_input, res_beta, res_alpha, quadrature)
    
    # focus loss
    loss_focus = -1 * jnp.log(output["focus_logits"][y["focus"]]) + logsumexp(output["focus_logits"])
    if output["stop"]:
        return loss_focus

    # atom type loss
    loss_type = -1 * jnp.log(output["atom_type_logits"][y["atom_type"]]) + logsumexp(output["atom_type_logits"])

    # position loss
    f = to_s2grid(output["position_coeffs"], res_beta, res_alpha, quadrature=quadrature)
    position_signal = position_signal.apply(jnp.exp)
    prob_radius = position_signal.integrate()
    true_eval =   # f(r*, r_hat*)
    loss_pos = -true_eval + jnp.log(jnp.sum(prob_radius) * radius_weights)

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
