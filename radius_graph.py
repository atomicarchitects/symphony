from typing import Tuple

import jax
import jax.numpy as jnp
import jraph
import matscipy.neighbours
import numpy as np


def get_edge_padding_mask(n_node: jnp.ndarray, n_edge: jnp.ndarray) -> jnp.ndarray:
    return jraph.get_edge_padding_mask(
        jraph.GraphsTuple(
            nodes=None,
            edges=None,
            globals=None,
            receivers=None,
            senders=n_edge,
            n_node=n_node,
            n_edge=n_edge,
        )
    )


def create_radius_graph(
    positions: jnp.ndarray, n_node: jnp.ndarray, cutoff: float, n_edge: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create a radius graph.

    Args:
        positions (jnp.ndarray): (n_node_sum, 3) array of positions, assumed to be padded, see `jraph.pad_with_graphs`
        n_node (jnp.ndarray): (n_graph,) array of number of nodes per graph, see `jraph.batch_np` and `jraph.pad_with_graphs`
        cutoff (float): cutoff radius
        n_edge (int): the total number of edges in output graph, the output graph will be padded with `jraph.pad_with_graphs`

    Returns:
        jnp.ndarray: senders
        jnp.ndarray: receivers
        jnp.ndarray: n_edge
        bool: ok
    """
    return jax.pure_callback(
        _create_radius_graph,
        (
            jnp.empty(n_edge, dtype=jnp.int32),  # senders
            jnp.empty(n_edge, dtype=jnp.int32),  # receivers
            jnp.empty(n_node.shape, dtype=jnp.int32),  # n_edge
            jnp.empty((), dtype=jnp.bool_),  # ok
        ),
        jax.lax.stop_gradient(positions),
        n_node,
        cutoff,
        n_edge,
    )


def _create_radius_graph(
    positions: np.ndarray, n_node: np.ndarray, cutoff: float, n_edge: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_node_sum = positions.shape[0]
    n_graph = n_node.shape[0]

    graph = jraph.GraphsTuple(
        nodes=positions.copy(),
        edges=None,
        globals=None,
        receivers=np.array([], dtype=np.int32),
        senders=np.array([], dtype=np.int32),
        n_node=n_node.copy(),
        n_edge=np.array([0] * n_graph, dtype=np.int32),
    )
    graph = jraph.unpad_with_graphs(graph)
    graphs = jraph.unbatch_np(graph)
    new_graphs = []
    for g in graphs:
        pos = g.nodes.astype(np.float64)
        senders, receivers = matscipy.neighbours.neighbour_list(
            "ij", positions=pos, cutoff=float(cutoff), cell=np.eye(3)
        )
        g = jraph.GraphsTuple(
            nodes=None,
            edges=None,
            globals=None,
            receivers=receivers,
            senders=senders,
            n_node=g.n_node,
            n_edge=np.array([len(senders)], dtype=np.int32),
        )
        new_graphs.append(g)

    graph = jraph.batch_np(new_graphs)
    ok = np.array(np.sum(graph.n_edge) <= n_edge, dtype=np.bool_)

    if not ok:
        dummy_graph = jraph.GraphsTuple(
            nodes=None,
            edges=None,
            globals=None,
            receivers=np.zeros(n_edge, dtype=np.int32),
            senders=np.zeros(n_edge, dtype=np.int32),
            n_node=np.array([n_node_sum] + [0] * (n_graph - 1), dtype=np.int32),
            n_edge=np.array([n_edge] + [0] * (n_graph - 1), dtype=np.int32),
        )
        return (
            dummy_graph.senders,
            dummy_graph.receivers,
            dummy_graph.n_edge,
            ok,
        )

    graph = jraph.pad_with_graphs(
        graph, n_node=n_node_sum, n_edge=n_edge, n_graph=n_graph
    )
    return (
        graph.senders,
        graph.receivers,
        graph.n_edge,
        ok,
    )
