"""Tests for flax.examples.ogbg_molpcba.models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import jraph

import models


class ModelsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.rngs = {
            "params": jax.random.PRNGKey(0),
            "dropout": jax.random.PRNGKey(1),
        }
        n_node = jnp.arange(3, 11)
        n_edge = jnp.arange(4, 12)
        total_n_node = jnp.sum(n_node)
        total_n_edge = jnp.sum(n_edge)
        n_graph = n_node.shape[0]
        feature_dim = 10
        self.graphs = jraph.GraphsTuple(
            n_node=n_node,
            n_edge=n_edge,
            senders=jnp.zeros(total_n_edge, dtype=jnp.int32),
            receivers=jnp.ones(total_n_edge, dtype=jnp.int32),
            nodes=jnp.ones((total_n_node, feature_dim)),
            edges=jnp.zeros((total_n_edge, feature_dim)),
            globals=jnp.zeros((n_graph, feature_dim)),
        )

    @parameterized.product(
        dropout_rate=[0.0, 0.5, 1.0], output_size=[50, 100], num_layers=[2]
    )
    def test_mlp(self, dropout_rate, output_size, num_layers):
        # Input definition.
        nodes = self.graphs.nodes

        # Model definition.
        mlp = models.MLP(
            feature_sizes=[output_size] * num_layers,
            dropout_rate=dropout_rate,
            activation=lambda x: x,
            deterministic=False,
            layer_norm=False,
        )
        nodes_after_mlp, _ = mlp.init_with_output(self.rngs, nodes)

        # Test that dropout actually worked.
        num_masked_entries = jnp.sum(nodes_after_mlp == 0)
        num_total_entries = jnp.size(nodes_after_mlp)
        self.assertLessEqual(
            num_masked_entries, (dropout_rate + 0.05) * num_total_entries
        )
        self.assertLessEqual(
            (dropout_rate - 0.05) * num_total_entries, num_masked_entries
        )

        # Test the shape of the output.
        self.assertEqual(nodes_after_mlp.shape[-1], output_size)

    @parameterized.parameters(
        {
            "latent_size": 5,
            "output_nodes_size": 15,
            "use_edge_model": True,
        },
        {
            "latent_size": 5,
            "output_nodes_size": 15,
            "use_edge_model": False,
        },
    )
    def test_graph_net(
        self, latent_size: int, output_nodes_size: int, use_edge_model: bool
    ):
        # Input definition.
        graphs = self.graphs
        num_nodes = jnp.sum(graphs.n_node)
        num_edges = jnp.sum(graphs.n_edge)
        num_graphs = graphs.n_node.shape[0]

        # Model definition.
        net = models.GraphNet(
            latent_size=latent_size,
            num_mlp_layers=2,
            message_passing_steps=2,
            output_nodes_size=output_nodes_size,
            use_edge_model=use_edge_model,
        )
        output, _ = net.init_with_output(self.rngs, graphs)

        # Output should be graph with the same topology, but a
        # different number of features.
        self.assertIsInstance(output, jraph.GraphsTuple)
        self.assertSequenceEqual(list(output.n_node), list(graphs.n_node))
        self.assertSequenceEqual(list(output.n_edge), list(graphs.n_edge))
        self.assertSequenceEqual(list(output.senders), list(graphs.senders))
        self.assertSequenceEqual(list(output.receivers), list(graphs.receivers))
        self.assertEqual(output.nodes.shape, (num_nodes, output_nodes_size))

    @parameterized.parameters(
        {"latent_size": 15, "output_nodes_size": 15},
        {"latent_size": 5, "output_nodes_size": 5},
    )
    def test_graph_mlp(self, latent_size: int, output_nodes_size: int):
        graphs = self.graphs
        num_nodes = jnp.sum(graphs.n_node)

        # Model definition.
        net = models.GraphMLP(
            latent_size=latent_size,
            num_mlp_layers=2,
            output_nodes_size=output_nodes_size,
        )
        output, _ = net.init_with_output(self.rngs, graphs)

        # Output should be graph with the same topology, but a
        # different number of features.
        self.assertIsInstance(output, jraph.GraphsTuple)
        self.assertSequenceEqual(list(output.n_node), list(graphs.n_node))
        self.assertSequenceEqual(list(output.n_edge), list(graphs.n_edge))
        self.assertSequenceEqual(list(output.senders), list(graphs.senders))
        self.assertSequenceEqual(list(output.receivers), list(graphs.receivers))
        self.assertSequenceEqual(
            list(output.edges.flatten()), list(graphs.edges.flatten())
        )
        self.assertEqual(output.nodes.shape, (num_nodes, output_nodes_size))


if __name__ == "__main__":
    absltest.main()
