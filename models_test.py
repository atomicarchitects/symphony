"""Tests for models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import e3nn_jax as e3nn
import jraph

import models


class ModelsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()

    def test_segment_sample_2D(self):
        species_probs = jnp.asarray(
            [
                [2 / 21, 2 / 21, 1 / 21, 3 / 21, 2 / 21],
                [2 / 21, 2 / 21, 1 / 21, 3 / 21, 2 / 21],
                [2 / 31, 2 / 31, 1 / 31, 3 / 31, 2 / 31],
                [1 / 31, 2 / 31, 1 / 31, 3 / 31, 2 / 31],
                [3 / 31, 2 / 31, 1 / 31, 3 / 31, 2 / 31],
            ]
        )
        n_node = jnp.asarray([2, 3])
        segment_ids = models.get_segment_ids(n_node, num_nodes=len(species_probs))

        # Normalize the probabilities to sum up for 1 over all nodes in each graph.
        species_probs_summed = jraph.segment_sum(
            species_probs.sum(axis=-1), segment_ids, num_segments=len(n_node)
        )
        species_probs = species_probs / species_probs_summed[segment_ids, None]

        samples = jnp.zeros_like(species_probs)
        num_samples = 10000
        rngs = jax.vmap(jax.random.PRNGKey)(jnp.arange(num_samples))
        node_indices, species_indices = jax.jit(
            jax.vmap(
                lambda rng: models.segment_sample_2D(
                    species_probs,
                    segment_ids,
                    num_segments=len(n_node),
                    rng=rng,
                )
            )
        )(rngs)
        samples = samples.at[node_indices, species_indices].add(1)
        samples /= num_samples

        np.testing.assert_allclose(samples, species_probs, atol=0.01)

    def test_segment_softmax_2D_with_stop(self):
        species_logits = jnp.asarray(
            [
                [jnp.log(2), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
                [jnp.log(2), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
                [jnp.log(2), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
                [jnp.log(1), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
                [jnp.log(3), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)],
            ]
        )
        n_node = jnp.asarray([2, 3])
        stop_logits = jnp.asarray([0.0, 0.0])
        segment_ids = models.get_segment_ids(n_node, num_nodes=len(species_logits))

        species_probs, stop_probs = models.segment_softmax_2D_with_stop(
            species_logits, stop_logits, segment_ids, num_segments=len(n_node)
        )
        expected_species_probs = jnp.asarray(
            [
                [2 / 21, 2 / 21, 1 / 21, 3 / 21, 2 / 21],
                [2 / 21, 2 / 21, 1 / 21, 3 / 21, 2 / 21],
                [2 / 31, 2 / 31, 1 / 31, 3 / 31, 2 / 31],
                [1 / 31, 2 / 31, 1 / 31, 3 / 31, 2 / 31],
                [3 / 31, 2 / 31, 1 / 31, 3 / 31, 2 / 31],
            ]
        )
        expected_stop_probs = jnp.asarray([1 / 21, 1 / 31])

        np.testing.assert_allclose(species_probs, expected_species_probs)
        np.testing.assert_allclose(stop_probs, expected_stop_probs)


if __name__ == "__main__":
    absltest.main()
