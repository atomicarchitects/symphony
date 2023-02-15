from typing import Iterator

import ase
import ase.io
import jax
import jax.numpy as jnp
import jraph
import matscipy.neighbours
import numpy as np

from datatypes import NodesInfo, TrainingGlobalsInfo, TrainingNodesInfo


def ase_atoms_to_jraph_graph(atoms: ase.Atoms, cutoff: float) -> jraph.GraphsTuple:
    receivers, senders = matscipy.neighbours.neighbour_list(
        quantities="ij", positions=atoms.positions, cutoff=cutoff, cell=np.eye(3)
    )

    return jraph.GraphsTuple(
        nodes=NodesInfo(atoms.positions, atoms.numbers),
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
    rng: jnp.ndarray,
    graph: jraph.GraphsTuple,
    atomic_numbers: jnp.ndarray,
    epsilon: float = 0.01,
) -> Iterator[jraph.GraphsTuple]:
    """Generative sequence for a molecular graph.

    Args:
        rng: The random number generator.
        graph: The molecular graph.
        atomic_numbers: The atomic numbers of the target species.
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

    node_species = jnp.searchsorted(atomic_numbers, graph.nodes.atomic_numbers)

    # pick a random initial node
    rng, k = jax.random.split(rng)
    first_node = jax.random.randint(k, shape=(), minval=0, maxval=n)

    min_dist = dist[graph.senders == first_node].min()
    targets = graph.receivers[
        (graph.senders == first_node) & (dist < min_dist + epsilon)
    ]

    species_probability = jnp.bincount(
        node_species[targets], length=len(atomic_numbers)
    ) / len(targets)

    # pick a random target
    rng, k = jax.random.split(rng)
    target = jax.random.choice(k, targets)

    nodes = TrainingNodesInfo(
        positions=graph.nodes.positions,
        species=node_species,
        focus_probability=jnp.ones(len(graph.nodes.positions)),
    )
    globals = TrainingGlobalsInfo(
        stop=jnp.array([False], dtype=bool),  # [1]
        target_specie_probability=species_probability[None],  # [1, n_species]
        target_specie=graph.nodes.atomic_numbers[target][None],  # [1]
        target_position=(
            graph.nodes.positions[target] - graph.nodes.positions[first_node]
        )[
            None
        ],  # [1, 3]
    )
    training_graph = graph._replace(nodes=nodes, globals=globals)

    yield subgraph(training_graph, jnp.array([first_node]))

    visited = jnp.array([first_node, target])

    for _ in range(n - 2):
        mask = jnp.isin(graph.senders, visited) & ~jnp.isin(graph.receivers, visited)

        min_dist = dist[mask].min()
        mask = mask & (dist < min_dist + epsilon)

        # focus_probability
        focus_probability = jnp.bincount(
            graph.senders[mask], length=len(graph.nodes.positions)
        )
        focus_probability = focus_probability / focus_probability.sum()

        # pick a random focus node
        rng, k = jax.random.split(rng)
        focus_node = jax.random.choice(
            k, len(graph.nodes.positions), p=focus_probability
        )

        # target_specie_probability
        targets = graph.receivers[(graph.senders == focus_node) & mask]
        target_specie_probability = jnp.bincount(
            node_species[targets], length=len(atomic_numbers)
        ) / len(targets)

        # pick a random target
        rng, k = jax.random.split(rng)
        target_node = jax.random.choice(k, targets)

        nodes = TrainingNodesInfo(
            positions=graph.nodes.positions,
            species=node_species,
            focus_probability=focus_probability,
        )
        globals = TrainingGlobalsInfo(
            stop=jnp.array([False], dtype=bool),  # [1]
            target_specie_probability=target_specie_probability[None],  # [1, n_species]
            target_specie=graph.nodes.atomic_numbers[target_node][None],  # [1]
            target_position=(
                graph.nodes.positions[target_node] - graph.nodes.positions[focus_node]
            )[
                None
            ],  # [1, 3]
        )
        training_graph = graph._replace(nodes=nodes, globals=globals)

        # move focus node to the beginning of the visited list and yield the subgraph
        visited = jnp.roll(visited, -jnp.where(visited == focus_node, size=1)[0][0])
        yield subgraph(training_graph, visited)

        visited = jnp.concatenate([visited, jnp.array([target_node])])

    assert len(visited) == n

    nodes = TrainingNodesInfo(
        positions=graph.nodes.positions,
        species=node_species,
        focus_probability=jnp.zeros((n,), dtype=float),
    )
    globals = TrainingGlobalsInfo(
        stop=jnp.array([True], dtype=bool),  # [1]
        target_specie_probability=jnp.zeros(
            (1, len(atomic_numbers)), dtype=float
        ),  # [1, n_species]
        target_specie=jnp.array([0], dtype=int),  # [1]
        target_position=jnp.zeros((1, 3)),  # [1, 3]
    )
    yield graph._replace(nodes=nodes, globals=globals)
