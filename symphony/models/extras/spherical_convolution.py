from typing import Callable

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp


def s2_activation(x, res_beta, res_alpha, max_ell, activation, p_val=None, p_arg=None):
    """Applies S2 activation function to input x."""
    x_grid = e3nn.to_s2grid(
        x,
        res_beta,
        res_alpha,
        quadrature="gausslegendre",
        p_val=p_val,
        p_arg=p_arg,
    )
    x_act = x_grid.apply(activation)
    x_prime = e3nn.from_s2grid(
        x_act, e3nn.Irreps.spherical_harmonics(max_ell)
    )  # [..., channels_in, (lmax + 1)**2]
    return x_prime


class SphericalConvolution(hk.Module):
    r"""E(3)-equivariant spherical convolution."""

    def __init__(
        self,
        res_beta: int,
        res_alpha: int,
        max_ell: int,
        channels_in: int,
        channels_out: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        h_init: hk.initializers.Initializer = hk.initializers.RandomNormal(),
        p_val: int = None,
        p_arg: int = None,
    ):
        """
        Args:
            res_beta (int): number of points on the sphere in the :math:`\theta` direction
            res_alpha (int): number of points on the sphere in the :math:`\phi` direction
            max_ell (int): maximum l
            channels_in (int)
            channels_out (int)
            activation: if None, no activation function is used.
            h_init (hk.initializers.Initializer): initializer for an array of spherical filters along `beta` angle, shape (..., res_beta)
            p_val (int, optional): parity of the value of the signal
            p_arg (int, optional): parity of the argument of the signal
        """
        super(SphericalConvolution, self).__init__()
        self.res_beta = res_beta
        self.res_alpha = res_alpha
        self.max_ell = max_ell
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.activation = e3nn.normalize_function(activation)
        self.h_init = h_init
        self.p_val = p_val
        self.p_arg = p_arg

    def __call__(
        self,
        x: e3nn.IrrepsArray,
    ) -> e3nn.IrrepsArray:
        """Compute the output of a spherical convolution. Assumes that all inputs are in fourier space.
        Args:
            x: input IrrepsArray of shape [..., channels_in, (x.irreps.lmax + 1)**2]
        Returns:
            IrrepsArray of features after interaction, shape [..., channels_out, (lmax + 1)**2]
        """
        x_prime = s2_activation(
            x,
            self.res_beta,
            self.res_alpha,
            self.max_ell,
            self.activation,
            self.p_val,
            self.p_arg,
        )

        h = hk.get_parameter(
            "h",
            shape=(self.channels_out, self.channels_in, self.res_beta),
            dtype=x.dtype,
            init=self.h_init,
        )
        h_prime = e3nn.legendre_transform_from_s2grid(
            h,
            self.max_ell,
            self.res_beta,
            quadrature="gausslegendre",
        )  # [channels_out, channels_in, lmax + 1]

        ls = jnp.arange(self.max_ell + 1)
        w = 2 * jnp.pi * jnp.sqrt(4 * jnp.pi / (2 * ls + 1)) * h_prime
        w_reshaped = jnp.repeat(
            w, 2 * ls + 1, axis=-1
        )  # [channels_out, channels_in, (lmax + 1)**2]
        y_prime = jnp.einsum(
            "...ik,jik->...jk", x_prime.array, w_reshaped
        )  # [..., channels_out, (lmax + 1)**2]

        return e3nn.IrrepsArray(x_prime.irreps, y_prime)
