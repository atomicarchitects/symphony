"""Definition of the GNN model."""

from typing import Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp
import jraph


def add_graphs_tuples(graphs: jraph.GraphsTuple,
                      other_graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Adds the nodes, edges and global features from other_graphs to graphs."""
  return graphs._replace(
      nodes=graphs.nodes + other_graphs.nodes,
      edges=graphs.edges + other_graphs.edges,
      globals=graphs.globals + other_graphs.globals)


class MLP(nn.Module):
  """A multi-layer perceptron."""

  feature_sizes: Sequence[int]
  dropout_rate: float = 0
  deterministic: bool = True
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  layer_norm: bool = True

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    x = inputs
    for size in self.feature_sizes:
      x = nn.Dense(features=size)(x)
      x = self.activation(x)
      x = nn.Dropout(
          rate=self.dropout_rate, deterministic=self.deterministic)(x)
      if self.layer_norm:
        x = nn.LayerNorm()(x)
    return x


class GraphMLP(nn.Module):
  """Applies an MLP to each node in the graph, with no message-passing."""

  latent_size: int
  num_mlp_layers: int
  output_nodes_size: int
  activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
  dropout_rate: float = 0
  layer_norm: bool = True
  deterministic: bool = True

  @nn.compact
  def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    def embed_node_fn(nodes):
        return MLP([self.latent_size * self.num_mlp_layers] + [self.output_nodes_size],
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic,
              activation=self.activation)(nodes)
    return jraph.GraphMapFeatures(
        embed_node_fn=embed_node_fn)(graphs)


class GraphNet(nn.Module):
  """A complete Graph Network model defined with Jraph."""

  latent_size: int
  num_mlp_layers: int
  message_passing_steps: int
  output_nodes_size: int
  dropout_rate: float = 0
  skip_connections: bool = True
  use_edge_model: bool = True
  layer_norm: bool = True
  deterministic: bool = True

  @nn.compact
  def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # We will first linearly project the original features as 'embeddings'.
    embedder = jraph.GraphMapFeatures(
        embed_node_fn=nn.Dense(self.latent_size),
        embed_edge_fn=nn.Dense(self.latent_size),
        embed_global_fn=nn.Dense(self.latent_size))
    processed_graphs = embedder(graphs)

    # Now, we will apply a Graph Network once for each message-passing round.
    mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
    for _ in range(self.message_passing_steps):
      if self.use_edge_model:
        update_edge_fn = jraph.concatenated_args(
            MLP(mlp_feature_sizes,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic))
      else:
        update_edge_fn = None

      update_node_fn = jraph.concatenated_args(
          MLP(mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic))
      update_global_fn = jraph.concatenated_args(
          MLP(mlp_feature_sizes,
              dropout_rate=self.dropout_rate,
              deterministic=self.deterministic))

      graph_net = jraph.GraphNetwork(
          update_node_fn=update_node_fn,
          update_edge_fn=update_edge_fn,
          update_global_fn=update_global_fn)

      if self.skip_connections:
        processed_graphs = add_graphs_tuples(
            graph_net(processed_graphs), processed_graphs)
      else:
        processed_graphs = graph_net(processed_graphs)

      if self.layer_norm:
        processed_graphs = processed_graphs._replace(
            nodes=nn.LayerNorm()(processed_graphs.nodes),
            edges=nn.LayerNorm()(processed_graphs.edges),
            globals=nn.LayerNorm()(processed_graphs.globals),
        )

    # We predict an embedding for each node.
    decoder = jraph.GraphMapFeatures(
        embed_node_fn=nn.Dense(self.output_nodes_size))
    processed_graphs = decoder(processed_graphs)

    return processed_graphs
