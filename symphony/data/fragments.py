import functools
from typing import Iterator

import jax
import jax.numpy as jnp
import jraph

from symphony import datatypes


def generate_fragments(
    rng: jnp.ndarray,
    graph: jraph.GraphsTuple,
    n_species: int,
    nn_tolerance: float = 0.01,
    max_radius: float = 2.03,
    mode: str = "nn",
    heavy_first: bool = False,
    beta_com: float = 0.0,
    species_mass: jnp.ndarray = jnp.asarray([1.008, 12.011, 14.007, 15.999, 18.998]),
) -> Iterator[datatypes.Fragments]:
    """Generative sequence for a molecular graph.

    Args:
        rng: The random number generator.
        graph: The molecular graph.
        n_species: The number of different species considered.
        nn_tolerance: Tolerance for the nearest neighbours.
        max_radius: The maximum distance of the focus-target
        mode:
        heavy_first: If true, the hydrogen atoms in the molecule will be placed last.
        beta_com: Inverse temperature value for the center of mass.

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

    # find all nodes that are hydrogens
    if heavy_first:
        hydrogens = jnp.where(graph.nodes.species == 0)[0]
    else:
        hydrogens = None

    # make fragments
    try:
        rng, visited_nodes, frag = _make_first_fragment(
            rng,
            graph,
            dist,
            n_species,
            nn_tolerance,
            max_radius,
            mode,
            heavy_first,
            hydrogens,
            beta_com,
            species_mass,
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
                heavy_first,
                hydrogens,
            )
            yield frag
    except ValueError:
        pass
    else:
        assert len(visited_nodes) == n

        yield _make_last_fragment(graph, n_species)


def _make_first_fragment(
    rng,
    graph,
    dist,
    n_species,
    nn_tolerance,
    max_radius,
    mode,
    heavy_first=False,
    hydrogens=None,
    beta_com=0.0,
    species_mass=jnp.asarray([1.008, 12.011, 14.007, 15.999, 18.998]),
):
    # get distances from center of mass
    com = jnp.average(
        graph.nodes.positions,
        axis=0,
        weights=jax.vmap(lambda x: species_mass[x])(graph.nodes.species),
    )
    distances_com = jnp.linalg.norm(graph.nodes.positions - com, axis=1)
    probs_com = jax.nn.softmax(-beta_com * distances_com**2)
    rng, k = jax.random.split(rng)
    if heavy_first and jnp.argwhere(graph.nodes.species != 0).squeeze(-1).shape[0] > 0:
        heavy_indices = jnp.argwhere(graph.nodes.species != 0).squeeze(-1)
        first_node = jax.random.choice(k, heavy_indices, p=probs_com[heavy_indices,])
    else:
        first_node = jax.random.choice(
            k, jnp.arange(0, len(graph.nodes.positions)), p=probs_com
        )

    if mode == "nn":
        first_node_mask = jnp.isin(graph.senders, first_node)
        # if there is more than one heavy atom, all heavy atoms are connected to
        # at least one other heavy atom, so this check is sufficient
        if (
            heavy_first
            and (first_node_mask & ~jnp.isin(graph.receivers, hydrogens)).sum() > 0
        ):
            min_dist = dist[
                first_node_mask & ~jnp.isin(graph.receivers, hydrogens)
            ].min()
        else:
            min_dist = dist[(graph.senders == first_node)].min()
        targets = graph.receivers[
            (graph.senders == first_node) & (dist < min_dist + nn_tolerance)
        ]
        del min_dist
    if mode == "radius":
        targets = graph.receivers[(graph.senders == first_node) & (dist < max_radius)]

    if len(targets) == 0:
        raise ValueError("No targets found.")

    if heavy_first:
        targets_heavy = targets[~jnp.isin(targets, hydrogens)]
        if len(targets_heavy) != 0:
            targets = targets_heavy

    species_probability = (
        jnp.zeros((graph.nodes.positions.shape[0], n_species))
        .at[first_node]
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
        stop=False,
    )

    visited = jnp.array([first_node, target])
    return rng, visited, sample


def _make_middle_fragment(
    rng,
    visited,
    graph,
    dist,
    n_species,
    nn_tolerance,
    max_radius,
    mode,
    heavy_first=False,
    hydrogens=None,
):
    n_nodes = len(graph.nodes.positions)
    senders, receivers = graph.senders, graph.receivers

    mask = jnp.isin(senders, visited) & ~jnp.isin(receivers, visited)

    if heavy_first:
        species = jax.vmap(lambda x: x != 0)(graph.nodes.species)
        if species.sum() > species[visited].sum():
            mask = (
                mask & ~jnp.isin(senders, hydrogens) & ~jnp.isin(receivers, hydrogens)
            )

    if mode == "nn":
        min_dist = dist[mask].min()
        mask = mask & (dist < min_dist + nn_tolerance)
        del min_dist
    if mode == "radius":
        mask = mask & (dist < max_radius)

    n = jnp.zeros((n_nodes, n_species))
    for focus_node in range(n_nodes):
        targets = receivers[(senders == focus_node) & mask]
        n = n.at[focus_node].set(
            jnp.bincount(graph.nodes.species[targets], length=n_species)
        )

    if jnp.sum(n) == 0:
        raise ValueError("No targets found.")

    target_species_probability = n / jnp.sum(n)

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
        target_species_probability,
        target_node,
        stop=False,
    )

    return rng, new_visited, sample


def _make_last_fragment(graph, n_species):
    n_nodes = len(graph.nodes.positions)
    return _into_fragment(
        graph,
        visited=jnp.arange(len(graph.nodes.positions)),
        focus_node=0,
        target_species_probability=jnp.zeros((n_nodes, n_species)),
        target_node=0,
        stop=True,
    )


def _into_fragment(
    graph,
    visited,
    focus_node,
    target_species_probability,
    target_node,
    stop,
):
    pos = graph.nodes.positions
    nodes = datatypes.FragmentsNodes(
        positions=pos,
        species=graph.nodes.species,
        focus_and_target_species_probs=target_species_probability,
    )
    globals = datatypes.FragmentsGlobals(
        stop=jnp.array([stop], dtype=bool),  # [1]
        target_species=graph.nodes.species[target_node][None],  # [1]
        target_positions=(pos[target_node] - pos[focus_node])[None],  # [1, 3]
    )
    graph = graph._replace(nodes=nodes, globals=globals)

    if stop:
        assert len(visited) == len(pos)
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
