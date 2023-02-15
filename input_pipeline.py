"""Implements the input pipeline for the molecular graph generation task."""

from typing import Iterator

import ase
import ase.io
import jax
import jax.numpy as jnp
import jraph
import matscipy.neighbours
import numpy as np
import random
import tensorflow as tf

from datatypes import NodesInfo, FragmentGlobals, FragmentNodes
from qm9 import load_qm9


def ase_atoms_to_jraph_graph(atoms: ase.Atoms, cutoff: float) -> jraph.GraphsTuple:
    """Converts an ASE atoms object to a Jraph graph."""
    receivers, senders = matscipy.neighbours.neighbour_list(
        quantities="ij", positions=atoms.positions, cutoff=cutoff, cell=np.eye(3)
    )

    return jraph.GraphsTuple(
        nodes=TrainingNodesInfo(atoms.positions, atoms.numbers),  # will have a better probability system later
        edges=None,
        globals=None,
        senders=senders,
        receivers=receivers,
        n_node=np.array([len(atoms)]),
        n_edge=np.array([len(senders)]),
    )


def subgraph(graph: jraph.GraphsTuple, nodes: np.ndarray) -> jraph.GraphsTuple:
    """Extract a subgraph from a graph.

    Args:
        graph: The graph to extract a subgraph from.
        nodes: The indices of the nodes to extract.

    Returns:
        The subgraph.
    """
    assert (
        len(graph.n_edge) == 1 and len(graph.n_node) == 1
    ), "Only single graphs supported."

    # Find all edges that connect to the nodes.
    edges = np.isin(graph.senders, nodes) & np.isin(graph.receivers, nodes)

    new_node_indices = -np.ones(graph.n_node[0], dtype=int)
    new_node_indices[nodes] = np.arange(len(nodes))

    return jraph.GraphsTuple(
        nodes=jax.tree_util.tree_map(lambda x: x[nodes], graph.nodes),
        edges=jax.tree_util.tree_map(lambda x: x[edges], graph.edges),
        globals=graph.globals,
        senders=new_node_indices[graph.senders[edges]],
        receivers=new_node_indices[graph.receivers[edges]],
        n_node=np.array([len(nodes)]),
        n_edge=np.array([np.sum(edges)]),
    )


def generative_sequence(
    rng: jnp.ndarray, graph: jraph.GraphsTuple, epsilon: float = 0.01
) -> Iterator[jraph.GraphsTuple]:
    """Returns an iterator for a generative sequence for a molecular graph.

    Args:
        rng: The random number generator.
        graph: The molecular graph.
        epsilon: Tolerance for the nearest neighbours.

    Returns:
        A generator that yields the next subgraph.
        - The globals are:
            - a boolean indicating whether the molecule is complete
            - the target position and atomic number
        - The last node is the focus node.
    """
    n = len(graph.nodes.positions)

    assert (
        len(graph.n_edge) == 1 and len(graph.n_node) == 1
    ), "Only single graphs supported."
    assert n >= 2, "Graph must have at least two nodes."

    vectors = (
        graph.nodes.positions[graph.receivers] - graph.nodes.positions[graph.senders]
    )
    dist = np.linalg.norm(vectors, axis=1)  # [n_edge]

    # pick a random initial node
    rng, k = jax.random.split(rng)
    first_node = jax.random.randint(k, shape=(), minval=0, maxval=n)

    min_dist = dist[graph.senders == first_node].min()
    targets = graph.receivers[
        (graph.senders == first_node) & (dist < min_dist + epsilon)
    ]

    # pick a random target
    rng, k = jax.random.split(rng)
    target = jax.random.choice(k, targets, shape=())
    target_specie = graph.nodes.atomic_numbers[target][None]

    globals = TrainingGlobalsInfo(
        stop=jnp.array([False], dtype=bool),  # [1]
        target_position=graph.nodes.positions[target][None],  # [1, 3]
        target_specie=target_specie,  # [1]
        target_specie_probability=jnp.vmap()
    )
    yield subgraph(graph, jnp.array([first_node]))._replace(globals=globals)

    visited = jnp.array([first_node, target])

    for _ in range(n - 2):
        mask = jnp.isin(graph.senders, visited) & ~jnp.isin(graph.receivers, visited)
        min_dist = dist[mask].min()

        maks = mask & (dist < min_dist + epsilon)
        i = jnp.where(maks)[0]

        # pick a random edge
        rng, k = jax.random.split(rng)
        edge = jax.random.choice(k, i, shape=())

        focus_node = graph.senders[edge]
        target_node = graph.receivers[edge]

        # move focus node to the beginning of the visited list
        visited = jnp.roll(visited, -jnp.where(visited == focus_node, size=1)[0][0])
        globals = TrainingGlobalsInfo(
            stop=jnp.array([False], dtype=bool),  # [1]
            target_position=graph.nodes.positions[target_node][None],  # [1, 3]
            target_specie=graph.nodes.atomic_numbers[target_node][None],  # [1]
            target_specie_probability=
        )
        yield subgraph(graph, visited)._replace(globals=globals)

        visited = jnp.concatenate([visited, jnp.array([target_node])])

    globals = TrainingGlobalsInfo(
        stop=jnp.array([True], dtype=bool),  # [1]
        target_position=jnp.zeros((1, 3)),  # [1, 3]
        target_specie=jnp.array([0], dtype=int),  # [1]
        target_specie_probability=
    )
    yield graph._replace(globals=globals)


import time
