from typing import Iterator, List

import ase
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from roundmantissa import ceil_mantissa

from dataset import ase_atoms_to_jraph_graph, generative_sequence
from dynamically_batch import dynamically_batch


def dataloader(
    rng: jnp.ndarray,
    molecules: List[ase.Atoms],
    atomic_numbers: jnp.ndarray,
    epsilon: float,
    cutoff: float,
) -> Iterator[jraph.GraphsTuple]:
    MAX_N_NODES = 100
    MAX_N_EDGES = 1000
    MAX_N_GRAPHS = 10

    graph_molecules = [
        ase_atoms_to_jraph_graph(molecule, cutoff) for molecule in molecules
    ]
    assert all([isinstance(graph, jraph.GraphsTuple) for graph in graph_molecules])

    for graphs in dynamically_batch(
        sample_iterator(rng, graph_molecules, atomic_numbers, epsilon),
        MAX_N_NODES,
        MAX_N_EDGES,
        MAX_N_GRAPHS,
    ):
        yield pad_graph_to_nearest_ceil_mantissa(
            graphs,
            n_mantissa_bits=2,
            n_max_nodes=MAX_N_NODES,
            n_max_edges=MAX_N_EDGES,
            n_max_graphs=MAX_N_GRAPHS,
        )


def sample_iterator(rng, graph_molecules, atomic_numbers, epsilon):
    SAMPLES_POOL_SIZE = 1024

    samples = []
    while True:
        while len(samples) < SAMPLES_POOL_SIZE:
            rng, k = jax.random.split(rng)
            i = jax.random.randint(k, (), 0, len(graph_molecules))

            rng, k = jax.random.split(rng)
            samples += list(
                generative_sequence(k, graph_molecules[i], atomic_numbers, epsilon)
            )
            assert all([isinstance(sample, jraph.GraphsTuple) for sample in samples])

        rng, k = jax.random.split(rng)
        i = jax.random.randint(k, (), 0, len(samples))
        yield samples.pop(i)


def pad_graph_to_nearest_ceil_mantissa(
    graphs_tuple: jraph.GraphsTuple,
    *,
    n_mantissa_bits: int = 2,
    n_min_nodes: int = 1,
    n_min_edges: int = 1,
    n_min_graphs: int = 1,
    n_max_nodes: int = np.iinfo(np.int32).max,
    n_max_edges: int = np.iinfo(np.int32).max,
    n_max_graphs: int = np.iinfo(np.int32).max,
) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
        7batch_sizedes --> 8 nodes (2^3)
        5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
        8 nodes --> 9 nodes
        3 graphs --> 4 graphs

    Args:
        graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
        A graphs_tuple batched to the nearest power of two.
    """
    n_nodes = graphs_tuple.n_node.sum()
    n_edges = len(graphs_tuple.senders)
    n_graphs = graphs_tuple.n_node.shape[0]

    pad_nodes_to = ceil_mantissa(n_nodes + 1, n_mantissa_bits)
    pad_edges_to = ceil_mantissa(n_edges, n_mantissa_bits)
    pad_graphs_to = ceil_mantissa(n_graphs + 1, n_mantissa_bits)

    pad_nodes_to = np.clip(pad_nodes_to, n_min_nodes, n_max_nodes)
    pad_edges_to = np.clip(pad_edges_to, n_min_edges, n_max_edges)
    pad_graphs_to = np.clip(pad_graphs_to, n_min_graphs, n_max_graphs)

    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )
