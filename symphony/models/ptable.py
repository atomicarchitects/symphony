import jax.numpy as jnp

'''Encodings of periodic table location information (groups, rows, blocks).'''

groups = jnp.array(
    [0, 17,] +
    [0, 1, 12, 13, 14, 15, 16, 17] * 2 +
    list(range(0, 18)) * 2 +
    [0, 1] + [2] * 15 + list(range(3, 18)) + 
    [0, 1] + [2] * 15 + list(range(3, 18))
)
rows = jnp.array(
    [0] * 2 +
    [1] * 8 +
    [2] * 8 +
    [3] * 18 +
    [4] * 18 +
    [5] * 32 +
    [6] * 32
)
# s = 0, p = 1, ...
blocks = jnp.array(
    [0] * 2 +
    [0] * 2 +                       [1] * 6 +
    [0] * 2 +                       [1] * 6 +
    [0] * 2 +            [2] * 10 + [1] * 6 +
    [0] * 2 +            [2] * 10 + [1] * 6 +
    [0] * 2 + [3] * 14 + [2] * 10 + [1] * 6 +
    [0] * 2 + [3] * 14 + [2] * 10 + [1] * 6
)