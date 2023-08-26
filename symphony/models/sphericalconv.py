from typing import Callable

import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp

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
        p_val: int = 1,
        p_arg: int = -1,
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
            x: input IrrepsArray indicating node features
        Returns:
            IrrepsArray of atom features after interaction
        """
        h = hk.get_parameter(
            "h",
            shape=(self.channels_out, self.channels_in, self.res_beta),
            dtype=x.dtype,
            init=self.h_init,
        )
        # Apply activation layer.
        x_grid = e3nn.to_s2grid(
            x,
            self.res_beta,
            self.res_alpha,
            quadrature="gausslegendre",
            p_val=self.p_val,
            p_arg=self.p_arg,
        )
        x_act = x_grid.apply(self.activation)
        x_prime = e3nn.from_s2grid(x_act, e3nn.Irreps.spherical_harmonics(self.max_ell))
        h_prime = e3nn.legendre_transform_from_s2grid(
            h,
            self.max_ell,
            self.res_beta,
            quadrature="gausslegendre",
        )
        w = (
            2
            * jnp.pi
            * jnp.sqrt(
                4
                * jnp.pi
                / (
                    2
                    * jnp.repeat(
                        jnp.arange(self.max_ell + 1).reshape(1, 1, self.max_ell + 1),
                        self.channels_out,
                        axis=0,
                    )
                    + 1
                )
            )
            * h_prime
        )
        w_reshaped = jnp.repeat(
            w, 2 * jnp.arange(self.max_ell + 1) + 1, axis=-1
        )  # [channels_out, channels_in, (lmax + 1)**2]
        y_prime = jnp.einsum('jk,ijk->ik', x_prime.array, w_reshaped)

        return e3nn.IrrepsArray(x_prime.irreps, y_prime)