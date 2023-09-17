from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.pmap, axis_name='i')
def normalize(x):
  return x / jax.lax.psum(x, 'i')

print(normalize(jnp.arange(4.)))