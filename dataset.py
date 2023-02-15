from functools import partial
from typing import Iterator

import ase
import ase.io
import jax
import jax.numpy as jnp
import jraph
import matscipy.neighbours
import numpy as np

from datatypes import NodesInfo, Fragment, FragmentGlobals, FragmentNodes


def ase_atoms_to_jraph_graph(
    atoms: ase.Atoms, atomic_numbers: np.ndarray, cutoff: float
) -> jraph.GraphsTuple:
    # Create edges
    receivers, senders = matscipy.neighbours.neighbour_list(
        quantities="ij", positions=atoms.positions, cutoff=cutoff, cell=np.eye(3)
    )

    # Get the species indices
    species = jnp.searchsorted(atomic_numbers, atoms.numbers)

    return jraph.GraphsTuple(
        nodes=NodesInfo(atoms.positions, species),
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


def generate_fragments(
    rng: jnp.ndarray,
    graph: jraph.GraphsTuple,
    n_species: int,
    epsilon: float = 0.01,
) -> Iterator[Fragment]:
    """Generative sequence for a molecular graph.

    Args:
        rng: The random number generator.
        graph: The molecular graph.
        atomic_numbers: The atomic numbers of the target species.
        epsilon: Tolerance for the nearest neighbours.

    Returns:
        A sequence of fragments.
    """
    n = len(graph.nodes.positions)

    assert (
        len(graph.n_edge) == 1 and len(graph.n_node) == 1
    ), "Only single graphs supported."
    assert n >= 2, "Graph must have at least two nodes."

    # compute edge distances
    dist = np.linalg.norm(
        graph.nodes.positions[graph.receivers] - graph.nodes.positions[graph.senders],
        axis=1,
    )  # [n_edge]

    rng, visited_nodes, frag = _make_first_fragment(
        rng, graph, dist, n_species, epsilon
    )
    yield frag

    for _ in range(n - 2):
        rng, visited_nodes, frag = _make_middle_fragment(
            rng, visited_nodes, graph, dist, n_species, epsilon
        )
        yield frag

    assert len(visited_nodes) == n

    yield _make_last_fragment(graph, n_species)


def _make_first_fragment(rng, graph, dist, n_species, epsilon):
    # pick a random initial node
    rng, k = jax.random.split(rng)
    first_node = jax.random.randint(
        k, shape=(), minval=0, maxval=len(graph.nodes.positions)
    )

    min_dist = dist[graph.senders == first_node].min()
    targets = graph.receivers[
        (graph.senders == first_node) & (dist < min_dist + epsilon)
    ]

    species_probability = _normalized_bitcount(graph.nodes.species[targets], n_species)

    # pick a random target
    rng, k = jax.random.split(rng)
    target = jax.random.choice(k, targets)

    sample = _into_fragment(
        graph,
        visited=jnp.array([first_node]),
        focus_probability=jnp.array([1.0]),
        focus_node=first_node,
        target_specie_probability=species_probability,
        target_node=target,
        stop=False,
    )

    visited = jnp.array([first_node, target])
    return rng, visited, sample


def _make_middle_fragment(rng, visited, graph, dist, n_species, epsilon):
    n_nodes = len(graph.nodes.positions)
    senders, receivers = graph.senders, graph.receivers

    mask = jnp.isin(senders, visited) & ~jnp.isin(receivers, visited)

    min_dist = dist[mask].min()
    mask = mask & (dist < min_dist + epsilon)

    focus_probability = _normalized_bitcount(senders[mask], n_nodes)

    # pick a random focus node
    rng, k = jax.random.split(rng)
    focus_node = jax.random.choice(k, n_nodes, p=focus_probability)

    # target_specie_probability
    targets = receivers[(senders == focus_node) & mask]
    target_specie_probability = _normalized_bitcount(
        graph.nodes.species[targets], n_species
    )

    # pick a random target
    rng, k = jax.random.split(rng)
    target_node = jax.random.choice(k, targets)

    new_visited = jnp.concatenate([visited, jnp.array([target_node])])

    sample = _into_fragment(
        graph,
        visited,
        focus_probability,
        focus_node,
        target_specie_probability,
        target_node,
        stop=False,
    )

    return rng, new_visited, sample


def _make_last_fragment(graph, n_species):
    return _into_fragment(
        graph,
        visited=jnp.arange(len(graph.nodes.positions)),
        focus_probability=jnp.zeros((len(graph.nodes.positions),)),
        focus_node=0,
        target_specie_probability=jnp.zeros((n_species,)),
        target_node=0,
        stop=True,
    )


def _into_fragment(
    graph,
    visited,
    focus_probability,
    focus_node,
    target_specie_probability,
    target_node,
    stop,
):
    nodes = FragmentNodes(
        positions=graph.nodes.positions,
        species=graph.nodes.species,
        focus_probability=focus_probability,
    )
    globals = FragmentGlobals(
        stop=jnp.array([stop], dtype=bool),  # [1]
        target_specie_probability=target_specie_probability[None],  # [1, n_species]
        target_specie=graph.nodes.species[target_node][None],  # [1]
        target_position=(
            graph.nodes.positions[target_node] - graph.nodes.positions[focus_node]
        )[
            None
        ],  # [1, 3]
    )
    graph = graph._replace(nodes=nodes, globals=globals)

    if stop:
        assert len(visited) == len(graph.nodes.positions)
        return graph
    else:
        # put focus node at the beginning
        visited = _move_first(visited, focus_node)

        # return subgraph
        return subgraph(graph, visited)


@jax.jit
def _move_first(xs, x):
    return jnp.roll(xs, -jnp.where(xs == x, size=1)[0][0])


@partial(jax.jit, static_argnums=(1,))
def _normalized_bitcount(xs, n: int):
    assert xs.ndim == 1
    return jnp.bincount(xs, length=n) / len(xs)
