import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from typing import Tuple, Union, Sequence

# adapted from https://github.com/VincentStimper/resampled-base-flows/tree/master

class ResamplingDistribution(distrax.Independent):
    def __init__(
            self,
            base_dist,
            num_param_mlp_layers,
            param_dims,
            num_tries,
            eps=1e-1,
        ):
        super().__init__(base_dist, reinterpreted_batch_ndims=0)
        self.num_param_mlp_layers = num_param_mlp_layers
        self.accept_network = hk.nets.MLP(
            [param_dims] * (self.num_param_mlp_layers-1) + [1],
            activation=jax.nn.sigmoid,
            activate_final=True,
            w_init=hk.initializers.RandomNormal(1e-4),
            b_init=hk.initializers.RandomNormal(1e-4),
        )
        self.Z = -1.
        self.num_tries = num_tries
        self.eps = eps
    
    def log_prob(self, value: jnp.ndarray, training=True) -> jnp.ndarray:
        log_p_dist = self._distribution.log_prob(value)
        accept_prob = self.accept_network(value)
        if training or self.Z < 0.:
            Z_batch = jnp.mean(self.accept_network(value))
            if self.Z < 0.:
                self.Z = Z_batch
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch
        alpha = (1 - self.Z) ** (self.num_tries - 1)
        log_p = jnp.log((1 - alpha) * accept_prob / self.Z + alpha) + log_p_dist
        return log_p

    def sample(self, seed: jax.random.PRNGKey, sample_shape: Union[int, Sequence[int]] = ()) -> jnp.ndarray:
        samples = jnp.ones((self.num_tries, sample_shape)) * jnp.nan
        def _try(t, seed):
            # is there anywhere we can get the size of the samples in this dist?
            # because it's not always 1 is it
            # yeah say it's 100 bc that's something i actually do at some point
            seed, sample_seed = jax.random.split(seed)
            z_ = self._distribution.sample(seed=sample_seed, sample_shape=sample_shape)
            accept_prob = self.accept_network(jnp.atleast_2d(z_))  # shape [1, 1]
            seed, dec_key = jax.random.split(seed)
            dec = (t == self.num_tries-1) | (jax.random.uniform(dec_key, accept_prob.shape) < accept_prob)
            return jax.lax.cond(
                dec.astype(int).sum(),
                lambda _: z_[
                    jnp.where(dec, size=1)[0]
                ],
                lambda _: jnp.array([jnp.nan]),
                operand=None
            )
        samples = jax.vmap(
            lambda t, s: _try(t, s)
        )(
            jnp.arange(self.num_tries),
            jax.random.split(seed, self.num_tries)
        ).squeeze()
        out = jnp.where(~jnp.isnan(samples), size=1)[0]
        # jax.debug.print("samples: {samples}", samples=samples)
        # jax.debug.print("sample: {sample}", sample=out)
        return samples[out]

    def _sample_n(self, key: jax.random.PRNGKey, n: int) -> jnp.ndarray:
        """See `Distribution._sample_n`."""
        # TODO change
        return self._distribution.sample(seed=key, sample_shape=n)

    # def sample_n_and_log_prob(
    #         self,
    #         key: jax.random.PRNGKey,
    #         n: int
    #     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     # TODO change
    #     # I *think* this corresponds to the forward function?
    #     sample = jnp.zeros(n)
    #     log_p_dist = jnp.zeros(n)
    #     n_samples = 0
    #     moving_avg_ct = 0
    #     Z_sum = 0
    #     # try sampling
    #     for t in range(self.num_tries):
    #         # is there anywhere we can get the size of the samples in this dist?
    #         # because it's not always 1 is it
    #         z_, log_prob_ = self._distribution.sample_and_log_prob(key, n)  # shape [n,]
    #         accept_prob = self.accept_network(jnp.atleast_2d(z_))  # shape [n, 1]
    #         if self.training or self.Z < 0.:
    #             Z_sum += jnp.sum(accept_prob)
    #             moving_avg_ct += n
    #         key, dec_key = jax.random.split(key)
    #         dec = jax.random.uniform(dec_key, accept_prob.shape) < accept_prob
    #         for j, dec_ in enumerate(dec[:, 0]):
    #             if dec_ or t == self.num_tries - 1:
    #                 sample[n_samples] = z_[j]
    #                 log_p_dist[n_samples] = log_prob_[j]
    #                 n_samples += 1
    #             if n_samples == n:
    #                 break
    #         if n_samples == n:
    #             break
    #     accept_prob = self.accept_network(sample)
    #     if self.training or self.Z < 0.:
    #         key, sample_key = jax.random.split(key)
    #         z_, _ = self.sample(sample_key)
    #         Z_batch = jnp.mean(self.accept_network(z_))
    #         Z_ = (Z_sum + Z_batch) / (moving_avg_ct + 1)
    #         if self.Z < 0.:
    #             self.Z = Z_
    #         else:
    #             self.Z = (1 - self.eps) * self.Z + self.eps * Z_
    #         Z = self.Z
    #     else:
    #         Z = self.Z
    #     alpha = (1 - Z) ** (self.num_tries - 1)
    #     log_p = jnp.log((1 - alpha) * accept_prob[:, 0] / Z + alpha) + log_p_dist
    #     return sample, log_p
