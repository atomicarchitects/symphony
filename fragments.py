import functools
from typing import Iterator

import jax
import jax.numpy as jnp
import jraph

import datatypes


def generate_fragments(
    rng: jnp.ndarray,
    graph: jraph.GraphsTuple,
    n_species: int,
    nn_tolerance: float = 0.01,
    max_radius: float = 2.03,
    mode: str = "nn",
) -> Iterator[datatypes.Fragments]:
    """Generative sequence for a molecular graph.

    Args:
        rng: The random number generator.
        graph: The molecular graph.
        atomic_numbers: The atomic numbers of the target species.
        nn_tolerance: Tolerance for the nearest neighbours.
        max_radius: The maximum distance of the focus-target

    Returns:
        A sequence of fragments.
    """
    assert mode in ["nn", "radius"]
    n = len(graph.nodes.positions)

    assert (
        len(graph.n_edge) == 1 and len(graph.n_node) == 1
    ), "Only single graphs supported."
    assert n >= 2, "Graph must have at least two nodes."

    # compute edge distances
    dist = jnp.linalg.norm(
        graph.nodes.positions[graph.receivers] - graph.nodes.positions[graph.senders],
        axis=1,
    )  # [n_edge]

    try:
        rng, visited_nodes, frag = _make_first_fragment(
            rng, graph, dist, n_species, nn_tolerance, max_radius, mode
        )
        yield frag

        for _ in range(n - 2):
            rng, visited_nodes, frag = _make_middle_fragment(
                rng,
                visited_nodes,
                graph,
                dist,
                n_species,
                nn_tolerance,
                max_radius,
                mode,
            )
            yield frag
    except ValueError:
        pass
    else:
        assert len(visited_nodes) == n

        yield _make_last_fragment(graph, n_species)


def _make_first_fragment(rng, graph, dist, n_species, nn_tolerance, max_radius, mode):
    # pick a random initial node
    rng, k = jax.random.split(rng)
    first_node = jax.random.randint(
        k, shape=(), minval=0, maxval=len(graph.nodes.positions)
    )

    if mode == "nn":
        min_dist = dist[graph.senders == first_node].min()
        targets = graph.receivers[
            (graph.senders == first_node) & (dist < min_dist + nn_tolerance)
        ]
        del min_dist
    if mode == "radius":
        targets = graph.receivers[(graph.senders == first_node) & (dist < max_radius)]

    if len(targets) == 0:
        raise ValueError("No targets found.")

    species_probability = (
        jnp.zeros((graph.nodes.positions.shape[0], n_species + 1))
        .at[first_node, :n_species]
        .set(_normalized_bitcount(graph.nodes.species[targets], n_species))
    )

    # pick a random target
    rng, k = jax.random.split(rng)
    target = jax.random.choice(k, targets)

    sample = _into_fragment(
        graph,
        visited=jnp.array([first_node]),
        focus_node=first_node,
        target_species_probability=species_probability,
        target_node=target,
    )

    visited = jnp.array([first_node, target])
    return rng, visited, sample


def _make_middle_fragment(
    rng, visited, graph, dist, n_species, nn_tolerance, max_radius, mode
):
    n_nodes = len(graph.nodes.positions)
    senders, receivers = graph.senders, graph.receivers

    mask = jnp.isin(senders, visited) & ~jnp.isin(receivers, visited)

    mask = mask & (dist < max_radius)

    # use max_radius to compute the stop probability:
    s = jnp.array(
        [1.0 if jnp.sum((senders == i) & mask) == 0 else 0.0 for i in visited]
    )

    if mode == "nn":
        min_dist = dist[mask].min()
        mask = mask & (dist < min_dist + nn_tolerance)
        del min_dist

    n = jnp.zeros((len(visited), n_species))
    for i in visited:
        targets = receivers[(senders == i) & mask]
        n = n.at[i].set(jnp.bincount(graph.nodes.species[targets], length=n_species))

    if jnp.sum(n) == 0:
        raise ValueError("No targets found.")

    # target_species_probability
    # last entry is the stop probability
    ts_pr = jnp.zeros((n_nodes, n_species + 1))
    ts_pr = ts_pr.at[visited, :n_species].set(n)
    ts_pr = ts_pr.at[visited, -1].set(s)
    ts_pr = ts_pr / jnp.sum(ts_pr)

    # pick a random focus node
    rng, k = jax.random.split(rng)
    focus_probability = _normalized_bitcount(senders[mask], n_nodes)
    focus_node = jax.random.choice(k, n_nodes, p=focus_probability)

    # pick a random target
    rng, k = jax.random.split(rng)
    targets = receivers[(senders == focus_node) & mask]
    target_node = jax.random.choice(k, targets)

    new_visited = jnp.concatenate([visited, jnp.array([target_node])])

    sample = _into_fragment(
        graph,
        visited,
        focus_node,
        ts_pr,
        target_node,
    )

    return rng, new_visited, sample


def _make_last_fragment(graph, n_species):
    n_nodes = len(graph.nodes.positions)

    ts_pr = jnp.zeros((n_nodes, n_species + 1))
    ts_pr = ts_pr.at[:, -1].set(1.0)
    ts_pr = ts_pr / jnp.sum(ts_pr)

    return _into_fragment(
        graph,
        visited=jnp.arange(len(graph.nodes.positions)),
        focus_node=0,
        target_species_probability=ts_pr,
        target_node=0,
    )


def _into_fragment(
    graph,
    visited,
    focus_node,
    target_species_probability,
    target_node,
):
    pos = graph.nodes.positions
    nodes = datatypes.FragmentsNodes(
        positions=pos,
        species=graph.nodes.species,
        target_species_probs=target_species_probability,
    )
    globals = datatypes.FragmentsGlobals(
        target_species=graph.nodes.species[target_node][None],  # [1]
        target_positions=(pos[target_node] - pos[focus_node])[None],  # [1, 3]
    )
    graph = graph._replace(nodes=nodes, globals=globals)

    if len(visited) == len(pos):
        return graph
    else:
        # put focus node at the beginning
        visited = _move_first(visited, focus_node)

        # return subgraph
        return subgraph(graph, visited)


@jax.jit
def _move_first(xs, x):
    return jnp.roll(xs, -jnp.where(xs == x, size=1)[0][0])


@functools.partial(jax.jit, static_argnums=(1,))
def _normalized_bitcount(xs, n: int):
    assert xs.ndim == 1
    return jnp.bincount(xs, length=n) / len(xs)


def subgraph(graph: jraph.GraphsTuple, nodes: jnp.ndarray) -> jraph.GraphsTuple:
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
    edges = jnp.isin(graph.senders, nodes) & jnp.isin(graph.receivers, nodes)

    new_node_indices = -jnp.ones(graph.n_node[0], dtype=int)
    new_node_indices = new_node_indices.at[nodes].set(jnp.arange(len(nodes)))
    # new_node_indices[nodes] = jnp.arange(len(nodes))

    return jraph.GraphsTuple(
        nodes=jax.tree_util.tree_map(lambda x: x[nodes], graph.nodes),
        edges=jax.tree_util.tree_map(lambda x: x[edges], graph.edges),
        globals=graph.globals,
        senders=new_node_indices[graph.senders[edges]],
        receivers=new_node_indices[graph.receivers[edges]],
        n_node=jnp.array([len(nodes)]),
        n_edge=jnp.array([jnp.sum(edges)]),
    )
