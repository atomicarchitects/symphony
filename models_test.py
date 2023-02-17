"""Tests for flax.examples.ogbg_molpcba.models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import haiku as hk

import models
import datatypes


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
        self.graphs = datatypes.Fragment(
            n_node=n_node,
            n_edge=n_edge,
            senders=jnp.zeros(total_n_edge, dtype=jnp.int32),
            receivers=jnp.ones(total_n_edge, dtype=jnp.int32),
            nodes=datatypes.FragmentNodes(
                positions=jnp.ones((total_n_node, 3)),
                species=(jnp.arange(total_n_node) % models.NUM_ELEMENTS),
                focus_probability=jnp.ones(total_n_node) / total_n_node,
            ),
            edges=jnp.zeros((total_n_edge, 10)),
            globals=datatypes.FragmentGlobals(
                stop=jnp.zeros((n_graph,)),
                target_positions=jnp.ones((n_graph, 3)),
                target_species=jnp.arange(n_graph) % models.NUM_ELEMENTS,
                target_species_probability=jnp.ones((n_graph, models.NUM_ELEMENTS))
                / models.NUM_ELEMENTS,
            ),
        )

    @parameterized.product(
        dropout_rate=[0.0, 0.5, 1.0], output_size=[50, 100], num_layers=[2]
    )
    def test_mlp(self, dropout_rate, output_size, num_layers):
        # Input definition.
        nodes = self.graphs.nodes.positions

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
            "use_edge_model": True,
        },
        {
            "latent_size": 5,
            "use_edge_model": False,
        },
    )
    def test_graph_net(self, latent_size: int, use_edge_model: bool):
        # Input definition.
        graphs = self.graphs
        num_nodes = jnp.sum(graphs.n_node)
        num_graphs = graphs.n_node.shape[0]

        # Model definition.
        net = models.GraphNet(
            latent_size=latent_size,
            num_mlp_layers=2,
            message_passing_steps=2,
            use_edge_model=use_edge_model,
        )
        output, _ = net.init_with_output(self.rngs, graphs)

        # Check that the shapes are all that we expect.
        self.assertIsInstance(output, datatypes.Predictions)
        self.assertSequenceEqual(output.focus_logits.shape, (num_nodes,))
        self.assertSequenceEqual(
            output.species_logits.shape, (num_graphs, models.NUM_ELEMENTS)
        )
        self.assertSequenceEqual(
            output.position_coeffs.shape[:2], (num_graphs, models.RADII.shape[0])
        )
        self.assertLen(output.position_coeffs.shape, 3)

    @parameterized.parameters(
        {"latent_size": 15},
        {"latent_size": 5},
    )
    def test_graph_mlp(self, latent_size: int):
        graphs = self.graphs
        num_nodes = jnp.sum(graphs.n_node)
        num_graphs = graphs.n_node.shape[0]

        # Model definition.
        net = models.GraphMLP(
            latent_size=latent_size,
            num_mlp_layers=2,
        )
        output, _ = net.init_with_output(self.rngs, graphs)

        # Check that the shapes are all that we expect.
        self.assertIsInstance(output, datatypes.Predictions)
        self.assertSequenceEqual(output.focus_logits.shape, (num_nodes,))
        self.assertSequenceEqual(
            output.species_logits.shape, (num_graphs, models.NUM_ELEMENTS)
        )
        self.assertSequenceEqual(
            output.position_coeffs.shape[:2], (num_graphs, models.RADII.shape[0])
        )
        self.assertLen(output.position_coeffs.shape, 3)

    @parameterized.parameters(
        {"latent_size": 15},
        {"latent_size": 5},
    )
    def test_haiku_graph_mlp(self, latent_size: int):
        graphs = self.graphs
        num_nodes = jnp.sum(graphs.n_node)
        num_graphs = graphs.n_node.shape[0]

        # Model definition.
        net = hk.transform(
            lambda graphs: models.HaikuGraphMLP(
                latent_size=latent_size,
                num_mlp_layers=2,
            )(graphs)
        )
        params = net.init(self.rngs["params"], graphs)
        output = net.apply(params, None, graphs)

        # Check that the shapes are all that we expect.
        self.assertIsInstance(output, datatypes.Predictions)
        self.assertSequenceEqual(output.focus_logits.shape, (num_nodes,))
        self.assertSequenceEqual(
            output.species_logits.shape, (num_graphs, models.NUM_ELEMENTS)
        )
        self.assertSequenceEqual(
            output.position_coeffs.shape[:2], (num_graphs, models.RADII.shape[0])
        )
        self.assertLen(output.position_coeffs.shape, 3)


if __name__ == "__main__":
    absltest.main()
