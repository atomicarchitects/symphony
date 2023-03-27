"""Tests for models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import haiku as hk

import models
import train_test


class ModelsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()

    def test_segment_sample(self):
        probs = jnp.asarray([0.1, 0.2, 0.7, 0.4, 0.6])
        segment_ids = jnp.asarray([0, 0, 0, 1, 1])
        samples = jnp.asarray([0, 0, 0, 0, 0])
        num_samples = 1000
        for seed in range(num_samples):
            sampled_indices = models.segment_sample(
                probs,
                segment_ids,
                num_segments=2,
                rng=jax.random.PRNGKey(seed),
            )
            samples = samples.at[sampled_indices].add(1)
        samples /= num_samples

        self.assertSequenceAlmostEqual(samples, probs, places=1)

    def test_segment_softmax_with_zero(self):
        logits = jnp.asarray(
            [jnp.log(2), jnp.log(2), jnp.log(1), jnp.log(3), jnp.log(2)]
        )
        n_node = jnp.asarray([2, 3])
        segment_ids = models.get_segment_ids(n_node, num_nodes=5, num_graphs=2)

        probs, zero_probs = models.segment_softmax_with_zero(
            logits, segment_ids, num_segments=2
        )
        expected_probs = jnp.asarray([2 / 4, 2 / 4, 1 / 6, 3 / 6, 2 / 6])
        expected_zero_probs = jnp.asarray([1 / 5, 1 / 7])

        self.assertSequenceAlmostEqual(probs, expected_probs, places=4)
        self.assertSequenceAlmostEqual(zero_probs, expected_zero_probs, places=4)


if __name__ == "__main__":
    absltest.main()
