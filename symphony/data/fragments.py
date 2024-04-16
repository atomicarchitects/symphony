from typing import Iterator, List

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import chex
from scipy.spatial.distance import pdist, squareform
import collections


from symphony import datatypes
from symphony.models import ptable


def get_connectivity(graph):
    # retrieve positions
    atom_positions = graph.nodes.positions
    # get pairwise distances (condensed)
    pairwise_distances = pdist(atom_positions)
    # use cutoff to obtain connectivity matrix
    connectivity = squareform(np.array(pairwise_distances <= 3.5, dtype=float))
    # set diagonal entries to zero (as we do not assume atoms to be their own neighbors)
    connectivity[np.diag_indices_from(connectivity)] = 0
    return connectivity


def is_connected(graph):
    con_mat = get_connectivity(graph)
    seen, queue = {0}, collections.deque([0])  # start at node (atom) 0
    while queue:
        vertex = queue.popleft()
        # iterate over (bonded) neighbors of current node
        for node in np.argwhere(con_mat[vertex] > 0).flatten():
            # add node to queue and list of seen nodes if it has not been seen before
            if node not in seen:
                seen.add(node)
                queue.append(node)
    # if the seen nodes do not include all nodes, there are disconnected parts
    return seen == {*range(len(con_mat))}


def generate_fragments(
    rng: chex.PRNGKey,
    graph: jraph.GraphsTuple,
    n_species: int,
    nn_tolerance: float = 0.01,
    max_radius: float = 2.03,
    mode: str = "nn",
    num_nodes_for_multifocus: int = 1,
    heavy_first: bool = False,
    beta_com: float = 0.0,
    max_targets_per_graph: int = 1,
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

    assert is_connected(graph)

    # compute edge distances
    dist = np.linalg.norm(
        graph.nodes.positions[graph.receivers] - graph.nodes.positions[graph.senders],
        axis=1,
    )  # [n_edge]

    # make fragments
    # try:
    rng, visited_nodes, frag = _make_first_fragment(
        rng,
        graph,
        dist,
        n_species,
        nn_tolerance,
        max_radius,
        mode,
        num_nodes_for_multifocus,
        heavy_first,
        beta_com,
        max_targets_per_graph=max_targets_per_graph,
    )
    yield frag

    counter = 0
    while counter < n - 2 and len(visited_nodes) < n:
        rng, visited_nodes, frag = _make_middle_fragment(
            rng,
            visited_nodes,
            graph,
            dist,
            n_species,
            nn_tolerance,
            max_radius,
            mode,
            num_nodes_for_multifocus,
            heavy_first,
            max_targets_per_graph=max_targets_per_graph,
        )
        yield frag
        counter += 1
    # except ValueError:
    #     pass
    # else:
    assert len(visited_nodes) == n
    yield _make_last_fragment(graph, n_species, max_targets_per_graph, num_nodes_for_multifocus)


def _make_first_fragment(
    rng,
    graph,
    dist,
    n_species,
    nn_tolerance,
    max_radius,
    mode,
    num_nodes_for_multifocus,
    heavy_first=False,
    beta_com=0.0,
    max_targets_per_graph: int = 1,
):
    # get distances from central transition metal - assume all atoms have the same mass
    n_nodes = len(graph.nodes.positions)
    bound1 = ptable.groups[graph.nodes.species] >= 2
    bound2 = ptable.groups[graph.nodes.species] <= 11
    com = np.average(graph.nodes.positions[bound1 & bound2], axis=0)
    distances_com = jnp.linalg.norm(graph.nodes.positions - com, axis=1)
    probs_com = jax.nn.softmax(-beta_com * distances_com**2)
    probs_com = jnp.where(bound1 & bound2, probs_com, 0.0)
    probs_com = probs_com / jnp.sum(probs_com)
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

    # pick a random target species
    rng, species_rng = jax.random.split(rng)
    target_species = jax.random.choice(species_rng, graph.nodes.species[targets]).reshape((1,))
    targets_of_same_species = targets[graph.nodes.species[targets] == target_species]
    # get all potential positions for that species
    rng, target_rng = jax.random.split(rng)
    target = jax.random.choice(target_rng, targets_of_same_species)
    target_positions = (
        graph.nodes.positions[targets_of_same_species]
        - graph.nodes.positions[first_node]
    )
    rng, k = jax.random.split(rng)
    target_positions = jax.random.permutation(k, target_positions)[
        :max_targets_per_graph
    ]
    target_positions_reshaped = np.zeros((num_nodes_for_multifocus, max_targets_per_graph, 3))
    target_positions_reshaped[0, : len(target_positions)] = target_positions
    target_mask = np.zeros((num_nodes_for_multifocus, max_targets_per_graph,))
    target_mask[0, : len(target_positions)] = 1

    sample = _into_fragment(
        graph,
        visited=np.array([first_node]),
        focus_mask=(np.arange(n_nodes) == first_node).astype(int),
        target_species_probability=species_probability,
        target_species=(np.arange(num_nodes_for_multifocus) == first_node).astype(int) * target_species,
        target_dist=target_positions_reshaped,
        target_mask=target_mask,
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
    num_nodes_for_multifocus,
    heavy_first=False,
    max_targets_per_graph: int = 1,
):
    n_nodes = len(graph.nodes.positions)
    senders, receivers = graph.senders, graph.receivers

    mask = jnp.isin(senders, visited) & ~jnp.isin(receivers, visited)
    assert sum(mask) > 0, err()

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
    focus_probability = _normalized_bitcount(senders[mask], n_nodes)
    if visited.sum() >= num_nodes_for_multifocus:
        focus_nodes = np.where(focus_probability > 0)[0]
        if focus_nodes.shape[0] > num_nodes_for_multifocus:
            focus_node_exclude = np.random.choice(focus_nodes, size=focus_nodes.shape[0] - num_nodes_for_multifocus, replace=False)
            focus_nodes = focus_nodes[~np.isin(focus_nodes, focus_node_exclude)]
    else:
        focus_nodes = np.asarray([np.argmax(focus_probability)])

    def choose_target_node(focus_node, key):
        """Picks a random target node for a given focus node."""
        mask_for_focus_node = (senders == focus_node) & mask
        target_edge_ndx = jax.random.choice(key, np.arange(receivers.shape[0]), p=mask_for_focus_node)
        return target_edge_ndx

    focus_mask = np.isin(np.arange(n_nodes), focus_nodes)

    # Pick the target nodes that maximize the number of unique targets.
    best_num_targets = 0
    best_target_ndxs = None
    for _ in range(10):
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, n_nodes)
        target_ndxs = jax.vmap(choose_target_node)(np.arange(n_nodes), keys)
        num_unique_targets = len(np.unique(target_ndxs[focus_nodes]))
        if num_unique_targets > best_num_targets:
            best_num_targets = num_unique_targets
            best_target_ndxs = target_ndxs

    target_ndxs = best_target_ndxs[focus_mask]
    target_nodes = receivers[target_ndxs]
    target_dist = np.zeros((num_nodes_for_multifocus, max_targets_per_graph, 3))
    target_mask = np.zeros((num_nodes_for_multifocus, max_targets_per_graph))
    target_species = np.zeros((num_nodes_for_multifocus,))
    target_species[:target_nodes.shape[0]] = graph.nodes.species[target_nodes]

    # Pick neighboring nodes of the same type as the given target node, per focus.
    focus_per_target = senders[target_ndxs]
    for i in range(target_ndxs.shape[0]):
        targets = receivers[(senders == focus_per_target[i]) & mask]
        targets_of_same_species = targets[graph.nodes.species[targets] == target_species[i]][:max_targets_per_graph]
        target_dist[
            i, : len(targets_of_same_species)
        ] = graph.nodes.positions[targets_of_same_species] - graph.nodes.positions[focus_per_target[i]]
        target_mask[i, : len(targets_of_same_species)] = 1

    new_visited = np.concatenate([visited, target_nodes])
    new_visited = np.unique(new_visited)

    sample = _into_fragment(
        graph,
        visited,
        focus_mask,
        target_species_probability,
        target_species,
        target_dist,
        target_mask,
        stop=False,
    )

    return rng, new_visited, sample


def _make_last_fragment(graph, n_species, max_targets_per_graph: int = 1, num_nodes_for_multifocus: int = 1):
    n_nodes = len(graph.nodes.positions)
    return _into_fragment(
        graph,
        visited=np.arange(len(graph.nodes.positions)),
        focus_mask=np.zeros((n_nodes,)),
        target_species_probability=np.zeros((n_nodes, n_species)),
        target_species=np.zeros((num_nodes_for_multifocus,)),
        target_dist=np.zeros((num_nodes_for_multifocus, max_targets_per_graph, 3)),
        target_mask=np.zeros((num_nodes_for_multifocus, max_targets_per_graph)),
        stop=True,
    )


def _into_fragment(
    graph,
    visited,
    focus_mask,
    target_species_probability,
    target_species,
    target_dist,
    target_mask,
    stop,
):

    nodes = datatypes.FragmentsNodes(
        positions=graph.nodes.positions,
        species=graph.nodes.species,
        focus_and_target_species_probs=target_species_probability,
        focus_mask=focus_mask,
    )
    globals = datatypes.FragmentsGlobals(
        stop=np.array([stop], dtype=bool),  # [1]
        target_species=np.expand_dims(target_species, axis=0),  # [1, num_nodes_for_multifocus]
        target_positions=np.expand_dims(target_dist, axis=0),  # [1, num_nodes_for_multifocus, max_targets_per_graph, 3]
        target_position_mask=np.expand_dims(target_mask, axis=0),  # [1, num_nodes_for_multifocus, max_targets_per_graph]
    )
    graph = graph._replace(nodes=nodes, globals=globals)

    if stop:
        assert len(visited) == len(graph.nodes.positions)
        return graph
    else:
        # # put focus node at the beginning
        # visited = _move_first(visited, focus_node)
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
