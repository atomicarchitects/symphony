from typing import Iterator

import jax
import jraph
import numpy as np
import chex

from symphony import datatypes


def debug_print(*args):
    return
    print(*args)


def generate_fragments(
    rng: chex.PRNGKey,
    graph: jraph.GraphsTuple,
    num_species: int,
    nn_tolerance: float = 0.01,
    max_radius: float = 2.03,
    mode: str = "nn",
    heavy_first: bool = False,
    beta_com: float = 0.0,
    num_nodes_lower_bound: int = 5,
) -> Iterator[datatypes.Fragments]:
    """Generative sequence for a molecular graph.

    Args:
        rng: The random number generator.
        graph: The molecular graph.
        num_species: The number of different species considered.
        nn_tolerance: Tolerance for the nearest neighbours.
        max_radius: The maximum distance of the focus-target
        mode: How to generate the fragments. Either "nn" or "radius".
        heavy_first: If true, the hydrogen atoms in the molecule will be placed last.
        beta_com: Inverse temperature value for the center of mass.
        num_nodes_lower_bound: How many nodes must be in the graph before multiple focuses are allowed.

    Returns:
        A sequence of fragments.
    """
    if mode not in ["nn", "radius"]:
        raise ValueError(f"Invalid mode: {mode}")

    if not (len(graph.n_edge) == 1 and len(graph.n_node) == 1):
        raise ValueError("Only single graphs supported.")

    num_nodes = len(graph.nodes.positions)
    if num_nodes < 2:
        raise ValueError("Graph must have at least two nodes.")

    debug_print("num_nodes", num_nodes)
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
            num_species,
            nn_tolerance,
            max_radius,
            mode,
            heavy_first,
            beta_com,
        )
        yield frag
        for _ in range(num_nodes - 2):
            rng, visited_nodes, frag = _make_middle_fragment(
                rng,
                visited_nodes,
                graph,
                dist,
                num_species,
                nn_tolerance,
                max_radius,
                mode,
                num_nodes_lower_bound,
                heavy_first,
            )
            yield frag

            if len(visited_nodes) == num_nodes:
                break
    except ValueError:
        raise
    else:
        assert len(visited_nodes) == num_nodes
        debug_print("visited_nodes", visited_nodes)
        yield _make_last_fragment(graph, num_species)


def _make_first_fragment(
    rng,
    graph,
    dist,
    num_species,
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
    num_nodes = len(graph.nodes.positions)
    rng, first_node_rng = jax.random.split(rng)
    if heavy_first and (graph.nodes.species != 0).sum() > 0:
        heavy_indices = np.argwhere(graph.nodes.species != 0).squeeze(-1)
        first_node = jax.random.choice(
            first_node_rng, heavy_indices, p=probs_com[heavy_indices]
        )
    else:
        first_node = jax.random.choice(first_node_rng, num_nodes, p=probs_com)
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

    target_species_probability = np.zeros((num_nodes, num_species))
    target_species_probability[first_node] = _normalized_bitcount(
        graph.nodes.species[targets], num_species
    )

    # Pick a random target.
    rng, target_rng = jax.random.split(rng)
    target = jax.random.choice(target_rng, targets)

    sample = _into_fragment(
        graph,
        visited=np.asarray([first_node]),
        focus_mask=np.isin(np.arange(num_nodes), [first_node]),
        target_species_probability=target_species_probability,
        target_nodes=np.where(np.arange(num_nodes) == first_node, target, 0),
        stop=False,
    )

    visited = np.asarray([first_node, target])
    return rng, visited, sample


def _make_middle_fragment(
    rng,
    visited,
    graph,
    dist,
    num_species,
    nn_tolerance,
    max_radius,
    mode,
    num_nodes_lower_bound: int,
    heavy_first=False,
):
    num_nodes = len(graph.nodes.positions)
    senders, receivers = graph.senders, graph.receivers

    mask = np.isin(senders, visited) & ~np.isin(receivers, visited)

    if heavy_first:
        heavy = graph.nodes.species > 0
        if heavy.sum() > heavy[visited].sum():
            mask = (
                mask
                & (graph.nodes.species[senders] > 0)
                & (graph.nodes.species[receivers] > 0)
            )

    if mode == "nn":
        min_dist = dist[mask].min()
        mask = mask & (dist < min_dist + nn_tolerance)
        del min_dist
    if mode == "radius":
        mask = mask & (dist < max_radius)

    counts = np.zeros((num_nodes, num_species))
    for focus_node in range(num_nodes):
        targets = receivers[(senders == focus_node) & mask]
        counts[focus_node] = np.bincount(
            graph.nodes.species[targets], minlength=num_species
        )
        counts[focus_node] /= np.maximum(np.sum(counts[focus_node]), 1)

    if np.sum(counts) == 0:
        raise ValueError("No targets found.")

    target_species_probability = counts

    # pick a random focus node
    focus_probability = _normalized_bitcount(senders[mask], num_nodes)
    debug_print("focus_probability", focus_probability)
    if num_nodes >= num_nodes_lower_bound:
        focus_nodes = np.where(focus_probability > 0)[0]
    else:
        focus_nodes = np.argmax(focus_probability)
    debug_print("focus_nodes", focus_nodes)

    # pick a random target for each focus node
    def choose_target_node(focus_node, key):
        return jax.random.choice(key, receivers, p=((senders == focus_node) & mask))

    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, num_nodes)
    target_nodes = jax.vmap(choose_target_node)(np.arange(num_nodes), keys)
    focus_mask = np.isin(np.arange(num_nodes), focus_nodes)
    true_target_nodes = target_nodes[focus_mask]
    debug_print("target_nodes", true_target_nodes)
    new_visited = np.concatenate([visited, true_target_nodes])
    new_visited = np.unique(new_visited)

    sample = _into_fragment(
        graph,
        visited=visited,
        focus_mask=focus_mask,
        target_species_probability=target_species_probability,
        target_nodes=target_nodes,
        stop=False,
    )

    return rng, new_visited, sample


def _make_last_fragment(graph, num_species: int):
    """Make the last fragment in the sequence."""
    num_nodes = len(graph.nodes.positions)
    return _into_fragment(
        graph,
        visited=np.arange(num_nodes),
        focus_mask=np.zeros(num_nodes, dtype=bool),
        target_species_probability=np.zeros((num_nodes, num_species)),
        target_nodes=np.zeros(num_nodes, dtype=np.int32),
        stop=True,
    )


def _into_fragment(
    graph,
    visited,
    focus_mask,
    target_species_probability,
    target_nodes,
    stop,
):
    """Convert the visited nodes into a fragment."""
    assert len(target_nodes) == len(focus_mask), (len(target_nodes), len(focus_mask))

    assert np.all(~np.isnan(target_species_probability)), target_species_probability

    pos = graph.nodes.positions
    nodes = datatypes.FragmentsNodes(
        positions=pos,
        species=graph.nodes.species,
        focus_mask=focus_mask,  # [num_nodes]
        target_species_probs=target_species_probability,  # [num_nodes, num_species]
        target_species=graph.nodes.species[target_nodes],  # [num_nodes]
        target_positions=(pos[target_nodes] - pos),  # [num_nodes, 3]
    )
    globals = datatypes.FragmentsGlobals(
        stop=np.array([stop], dtype=bool),  # [1]
    )
    graph = graph._replace(nodes=nodes, globals=globals)

    if stop:
        assert len(visited) == len(pos)
        return graph
    else:
        # return subgraph
        debug_print("visited", visited)
        return subgraph(graph, visited)


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
