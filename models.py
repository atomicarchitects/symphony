"""Definition of the GNN model."""

from typing import Callable, Sequence, Union, Optional, Tuple

import e3nn_jax as e3nn
from flax import linen as nn
import jax.numpy as jnp
import jraph


class S2Activation(nn.Module):
    """Applies a non-linearity after projecting the signal to the sphere."""

    irreps: e3nn.Irreps
    resolution: Union[int, Tuple[int, int]]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    lmax_out: Optional[int] = None
    layer_norm: bool = True

    def _complete_lmax_and_res(
        lmax: Optional[int], res_beta: Optional[int], res_alpha: Optional[int]
    ) -> Tuple[int, int, int]:
        """Fills in the missing values for lmax, res_beta and res_alpha for e3nn.to_s2grid().

        To use FFT accurately, we would want:
            2 * (lmax) + 1 == res_alpha
            2 * (lmax + 1) == res_beta
        """
        if all(arg is None for arg in [lmax, res_beta, res_alpha]):
            raise ValueError("All the entries are None.")

        if res_beta is None:
            if lmax is not None:
                res_beta = 2 * (lmax + 1)  # minimum req. to go on sphere and back
            elif res_alpha is not None:
                res_beta = 2 * ((res_alpha + 1) // 2)

        if res_alpha is None:
            if lmax is not None:
                if res_beta is not None:
                    res_alpha = max(2 * lmax + 1, res_beta - 1)
                else:
                    res_alpha = 2 * lmax + 1  # minimum req. to go on sphere and back
            elif res_beta is not None:
                res_alpha = res_beta - 1

        if lmax is None:
            lmax = min(
                res_beta // 2 - 1, (res_alpha - 1) // 2
            )  # maximum possible to go on sphere and back

        assert res_beta % 2 == 0
        assert lmax + 1 <= res_beta // 2

        return lmax, res_beta, res_alpha

    def _extract_irreps_info(self) -> Tuple[int, int, int, int]:
        """Extracts information about the irreps and resolution of the input and output."""

        irreps = e3nn.Irreps(self.irreps).simplify()
        _, (lmax, _) = irreps[-1]

        assert all(mul == 1 for mul, _ in irreps)
        assert irreps.ls == list(range(lmax + 1))

        # The input transforms as : A_l ---> p_val * (p_arg)^l * A_l
        # The sphere signal transforms as : f(r) ---> p_val * f(p_arg * r)
        if self.lmax_out is None:
            lmax_out = lmax

        try:
            lmax, res_beta, res_alpha = self._complete_lmax_and_res(lmax, *self.res)
        except TypeError:
            lmax, res_beta, res_alpha = self._complete_lmax_and_res(
                lmax, self.res, None
            )

        return lmax, res_beta, res_alpha, lmax_out

    @nn.compact
    def __call__(self, feature_coeffs: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        (
            lmax,
            res_beta,
            res_alpha,
            lmax_out,
        ) = self._extract_irreps_info()
        assert feature_coeffs.irreps == self.irreps
        features = e3nn.to_s2grid(
            feature_coeffs,
            res_beta,
            res_alpha,
            quadrature="gausslegendre",
        )
        features = features.apply(self.activation)
        if self.layer_norm:
            features = nn.LayerNorm()(features)
        updated_feature_coeffs = e3nn.from_s2grid(
            features,
            lmax_out,
            lmax_in=lmax,
        )
        return updated_feature_coeffs


def add_graphs_tuples(
    graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(
        nodes=graphs.nodes + other_graphs.nodes,
        edges=graphs.edges + other_graphs.edges,
        globals=graphs.globals + other_graphs.globals,
    )


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
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
        return x


class S2MLP(nn.Module):
    """A E(3)-equivariant MLP with S2 activations."""

    layers_irreps_out: Sequence[e3nn.Irreps]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x
    skip_connections: bool = False
    s2_grid_resolution: Union[int, Tuple[int, int]] = 100

    @nn.compact
    def __call__(self, inputs: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        for index, irreps_out in enumerate(self.layers_irreps_out):
            # Apply linear layer.
            next_inputs = e3nn.flax.Linear(irreps_out)(inputs)

            # Apply activation.
            all_irreps = e3nn.Irreps(
                [(1, (l, -1)) for l in range(1 + next_inputs.irreps.lmax)]
            )
            next_inputs = e3nn.flax.Linear(all_irreps)(next_inputs)
            next_inputs = S2Activation(
                next_inputs.irreps, self.activation, self.s2_grid_resolution
            )(next_inputs)

            # Add skip connection.
            if self.skip_connections:
                next_inputs = e3nn.concatenate([next_inputs, inputs])

            inputs = next_inputs
        return inputs


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
            return MLP(
                [self.latent_size * self.num_mlp_layers] + [self.output_nodes_size],
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
                activation=self.activation,
            )(nodes)

        return jraph.GraphMapFeatures(embed_node_fn=embed_node_fn)(graphs)


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
            embed_global_fn=nn.Dense(self.latent_size),
        )
        processed_graphs = embedder(graphs)

        # Now, we will apply a Graph Network once for each message-passing round.
        mlp_feature_sizes = [self.latent_size] * self.num_mlp_layers
        for _ in range(self.message_passing_steps):
            if self.use_edge_model:
                update_edge_fn = jraph.concatenated_args(
                    MLP(
                        mlp_feature_sizes,
                        dropout_rate=self.dropout_rate,
                        deterministic=self.deterministic,
                    )
                )
            else:
                update_edge_fn = None

            update_node_fn = jraph.concatenated_args(
                MLP(
                    mlp_feature_sizes,
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                )
            )
            update_global_fn = jraph.concatenated_args(
                MLP(
                    mlp_feature_sizes,
                    dropout_rate=self.dropout_rate,
                    deterministic=self.deterministic,
                )
            )

            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                update_global_fn=update_global_fn,
            )

            if self.skip_connections:
                processed_graphs = add_graphs_tuples(
                    graph_net(processed_graphs), processed_graphs
                )
            else:
                processed_graphs = graph_net(processed_graphs)

            if self.layer_norm:
                processed_graphs = processed_graphs._replace(
                    nodes=nn.LayerNorm()(processed_graphs.nodes),
                    edges=nn.LayerNorm()(processed_graphs.edges),
                    globals=nn.LayerNorm()(processed_graphs.globals),
                )

        # We predict an embedding for each node.
        decoder = jraph.GraphMapFeatures(embed_node_fn=nn.Dense(self.output_nodes_size))
        processed_graphs = decoder(processed_graphs)

        return processed_graphs
