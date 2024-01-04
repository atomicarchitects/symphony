from typing import Iterator

import jax
import jraph
import numpy as np
import chex

from symphony import datatypes


def generate_fragments(
    rng: chex.PRNGKey,
    graph: jraph.GraphsTuple,
    n_species: int,
    nn_tolerance: float = 0.01,
    max_radius: float = 2.03,
    mode: str = "nn",
    heavy_first: bool = False,
    beta_com: float = 0.0,
) -> Iterator[datatypes.Fragments]:
    """Generative sequence for a molecular graph.

    Args:
        rng: The random number generator.
        graph: The molecular graph.
        n_species: The number of different species considered.
        nn_tolerance: Tolerance for the nearest neighbours.
        max_radius: The maximum distance of the focus-target
        mode: How to generate the fragments. Either "nn" or "radius".
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
    dist = np.linalg.norm(
        graph.nodes.positions[graph.receivers] - graph.nodes.positions[graph.senders],
        axis=1,
    )  # [n_edge]

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
            beta_com,
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
            )
            yield frag
    except ValueError:
        pass
    else:
        assert len(visited_nodes) == n

        yield _make_last_fragment(graph, n_species)


def generate_silica_fragments(
    rng,
    graph,
    atomic_numbers,
    nn_tolerance,
    max_radius,
    mode,
    heavy_first=False
):
    '''Removes a SiO4 tetrahedron from a silica-based structure and generates fragments from the result.'''
    n_species = len(atomic_numbers)
    n_nodes = graph.n_node[0]
    # compute edge distances
    dist = np.linalg.norm(
        graph.nodes.positions[graph.receivers] - graph.nodes.positions[graph.senders],
        axis=1,
    )  # [n_edge]
    for vec in graph.globals.cell:
        for mul in [1, -1]:
            dist = np.minimum(dist, np.linalg.norm(
                graph.nodes.positions[graph.receivers] - mul * vec - graph.nodes.positions[graph.senders],
                axis=1,
            ))
    for i in range(n_nodes):
        if graph.nodes.species[i] == 0:
            continue
        # find the closest O to this Si atom
        min_dist = 3.0
        ndx_exclude = [i]
        for j in range(n_nodes):
            if graph.nodes.species[j] == 1:
                continue
            # if abs(sum((graph.nodes.positions[j] - graph.nodes.positions[i])**2) - min_dist) < eps:
            if sum((graph.nodes.positions[j] - graph.nodes.positions[i])**2) < min_dist:
                ndx_exclude.append(j)
            for vec in graph.globals.cell:
                if sum((graph.nodes.positions[j] - vec - graph.nodes.positions[i])**2) < min_dist or sum((graph.nodes.positions[j] + vec - graph.nodes.positions[i])**2) < min_dist:
                    ndx_exclude.append(j)
        ndx_exclude = np.asarray(ndx_exclude)

        # use middle- and last-fragment generators; everything not in ndx_exclude can be "visited"
        visited_nodes = np.asarray(range(n_nodes))[~np.isin(np.asarray(range(n_nodes)), ndx_exclude)]

        # make fragments
        try:
            for _ in range(len(ndx_exclude)):
                rng, visited_nodes, frag = _make_middle_fragment(
                    rng,
                    visited_nodes,
                    graph,
                    dist,
                    n_species,
                    nn_tolerance,
                    max_radius,
                    mode,
                    periodic=True,
                    heavy_first=heavy_first
                )
                yield frag
        except ValueError:
            print("fragment could not be generated")
            pass
        else:
            assert len(visited_nodes) == n_nodes

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
    beta_com=0.0,
):
    # get distances from (approximate) center of mass - assume all atoms have the same mass
    com = np.average(
        graph.nodes.positions,
        axis=0,
        weights=(graph.nodes.species > 0) if heavy_first else None,
    )
    distances_com = np.linalg.norm(graph.nodes.positions - com, axis=1)
    probs_com = jax.nn.softmax(-beta_com * distances_com**2)
    rng, k = jax.random.split(rng)
    if heavy_first and (graph.nodes.species != 0).sum() > 0:
        heavy_indices = np.argwhere(graph.nodes.species != 0).squeeze(-1)
        first_node = jax.random.choice(k, heavy_indices, p=probs_com[heavy_indices])
    else:
        first_node = jax.random.choice(
            k, np.arange(0, len(graph.nodes.positions)), p=probs_com
        )
    first_node = int(first_node)

    mask = graph.senders == first_node
    if heavy_first and (mask & graph.nodes.species[graph.receivers] > 0).sum() > 0:
        mask = mask & (graph.nodes.species[graph.receivers] > 0)
    if mode == "nn":
        min_dist = dist[mask].min()
        targets = graph.receivers[mask & (dist < min_dist + nn_tolerance)]
        del min_dist
    if mode == "radius":
        targets = graph.receivers[mask & (dist < max_radius)]

    if len(targets) == 0:
        raise ValueError("No targets found.")

    species_probability = np.zeros((graph.nodes.positions.shape[0], n_species))
    species_probability[first_node] = _normalized_bitcount(
        graph.nodes.species[targets], n_species
    )

    # pick a random target
    rng, k = jax.random.split(rng)
    target = jax.random.choice(k, targets)

    sample = _into_fragment(
        graph,
        visited=np.array([first_node]),
        focus_node=first_node,
        target_species_probability=species_probability,
        target_node=target,
        stop=False,
    )

    visited = np.array([first_node, target])
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
    periodic=False
):
    n_nodes = len(graph.nodes.positions)
    senders, receivers = graph.senders, graph.receivers

    mask = np.isin(senders, visited) & ~np.isin(receivers, visited)

    if heavy_first:
        if np.sum(mask & graph.nodes.species[receivers] > 0) > 0:
            mask = mask & (graph.nodes.species[receivers] > 0)
        if np.sum(mask & (graph.nodes.species[senders] > 0)) > 0:
            mask = mask & (graph.nodes.species[senders] > 0)

    if mode == "nn":
        min_dist = dist[mask].min()
        mask = mask & (dist < min_dist + nn_tolerance)
        del min_dist
    if mode == "radius":
        mask = mask & (dist < max_radius)

    counts = np.zeros((n_nodes, n_species))
    for focus_node in range(n_nodes):
        targets = receivers[(senders == focus_node) & mask]
        counts[focus_node] = np.bincount(
            graph.nodes.species[targets], minlength=n_species
        )

    if np.sum(counts) == 0:
        raise ValueError("No targets found.")

    target_species_probability = counts / np.sum(counts)

    # pick a random focus node
    rng, k = jax.random.split(rng)
    focus_probability = _normalized_bitcount(senders[mask], n_nodes)
    focus_node = jax.random.choice(k, n_nodes, p=focus_probability)
    focus_node = int(focus_node)

    # pick a random target
    rng, k = jax.random.split(rng)
    targets = receivers[(senders == focus_node) & mask]
    target_node = jax.random.choice(k, targets)
    target_node = int(target_node)

    new_visited = np.concatenate([visited, np.array([target_node])])

    sample = _into_fragment(
        graph,
        visited,
        focus_node,
        target_species_probability,
        target_node,
        stop=False,
        periodic=periodic
    )

    return rng, new_visited, sample


def _make_last_fragment(graph, n_species):
    n_nodes = len(graph.nodes.positions)
    return _into_fragment(
        graph,
        visited=np.arange(len(graph.nodes.positions)),
        focus_node=0,
        target_species_probability=np.zeros((n_nodes, n_species)),
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
    periodic=False,
):
    pos = graph.nodes.positions
    target_positions = pos[target_node] - pos[focus_node]
    relative_positions = graph.nodes.positions[graph.receivers] - graph.nodes.positions[graph.senders]
    if periodic:
        # for periodic structures, re-center the target positions and recompute relative positions if necessary
        for vec in graph.globals.cell[0]:
            for d in [-1, 1]:
                if np.linalg.norm(pos[target_node] + vec * d - pos[focus_node]) < np.linalg.norm(target_positions):
                    target_positions = pos[target_node] + vec * d - pos[focus_node]
                shifted_rel_pos = graph.nodes.positions[graph.receivers] - graph.nodes.positions[graph.senders] + vec * d
                relative_positions = np.where(
                    np.linalg.norm(shifted_rel_pos, axis=-1).reshape(-1, 1) < np.linalg.norm(relative_positions, axis=-1).reshape(-1, 1),
                    shifted_rel_pos,
                    relative_positions)
    nodes = datatypes.FragmentsNodes(
        positions=pos,
        species=graph.nodes.species,
        focus_and_target_species_probs=target_species_probability,
    )
    globals = datatypes.FragmentsGlobals(
        stop=np.array([stop], dtype=bool),  # [1]
        target_species=graph.nodes.species[target_node][None],  # [1]
        target_positions=target_positions[None],  # [1, 3]
        cell=graph.globals.cell
    )
    edges = datatypes.FragmentsEdges(relative_positions=relative_positions)
    graph = graph._replace(nodes=nodes, edges=edges, globals=globals)

    if stop:
        assert len(visited) == len(pos)
        return graph
    else:
        # put focus node at the beginning
        visited = _move_first(visited, focus_node)
        visited = np.asarray(visited)

        # return subgraph
        return subgraph(graph, visited)


def _move_first(xs, x):
    return np.roll(xs, -np.where(xs == x)[0][0])


def _normalized_bitcount(xs, n: int):
    assert xs.ndim == 1
    return np.bincount(xs, minlength=n) / len(xs)


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
