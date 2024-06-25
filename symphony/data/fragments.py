from typing import Iterator, Optional

import jax
import jraph
import numpy as np
import chex

from symphony import datatypes
from symphony.models.ptable import PeriodicTableEmbedder


def generate_fragments(
    rng: chex.PRNGKey,
    graph: jraph.GraphsTuple,
    num_species: int,
    nn_tolerance: Optional[float],
    max_radius: Optional[float],
    mode: str,
    heavy_first: bool,
    max_targets_per_graph: int,
    transition_first: bool,  # TODO currently only handles structures w 1 transition metal
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
        max_targets_per_graph: The maximum number of targets per graph.

    Returns:
        A sequence of fragments.
    """
    if mode not in ["nn", "radius"]:
        raise ValueError("mode must be either 'nn' or 'radius'.")
    if mode == "radius" and max_radius is None:
        raise ValueError("max_radius must be specified for mode 'radius'.")
    if mode != "radius" and max_radius is not None:
        raise ValueError("max_radius specified, but mode is not 'radius'.")
    if mode == "nn" and nn_tolerance is None:
        raise ValueError("nn_tolerance must be specified for mode 'nn'.")
    if mode != "nn" and nn_tolerance is not None:
        raise ValueError("nn_tolerance specified, but mode is not 'nn'.")

    n = len(graph.nodes.positions)
    assert (
        len(graph.n_edge) == 1 and len(graph.n_node) == 1
    ), "Only single graphs supported."
    assert n >= 2, "Graph must have at least two nodes."

    # Compute edge distances.
    dist = np.linalg.norm(
        graph.nodes.positions[graph.receivers] - graph.nodes.positions[graph.senders],
        axis=1,
    )  # [n_edge]

    with jax.default_device(jax.devices("cpu")[0]):
        rng, visited_nodes, frag = _make_first_fragment(
            rng,
            graph,
            dist,
            num_species,
            nn_tolerance,
            max_radius,
            mode,
            heavy_first,
            max_targets_per_graph,
            transition_first,
        )
        yield frag

        for _ in range(n - 2):
            rng, visited_nodes, frag = _make_middle_fragment(
                rng,
                visited_nodes,
                graph,
                dist,
                num_species,
                nn_tolerance,
                max_radius,
                mode,
                heavy_first,
                max_targets_per_graph,
            )
            yield frag

        assert len(visited_nodes) == n
        yield _make_last_fragment(graph, num_species, max_targets_per_graph)


def pick_targets(
    rng,
    targets,
    node_species,
    target_species_probability_for_focus,
    max_targets_per_graph,
):
    # Pick a random target species.
    rng, k = jax.random.split(rng)
    target_species = jax.random.choice(
        k,
        len(target_species_probability_for_focus),
        p=target_species_probability_for_focus,
    )

    # Pick up to max_targets_per_graph targets of the target species.
    targets_of_this_species = targets[node_species[targets] == target_species]
    targets_of_this_species = targets_of_this_species[:max_targets_per_graph]

    return targets_of_this_species


def _make_first_fragment(
    rng,
    graph,
    dist,
    num_species,
    nn_tolerance,
    max_radius,
    mode,
    heavy_first,
    max_targets_per_graph,
    transition_first,
):
    rng, k = jax.random.split(rng)
    if transition_first:
        ptable = PeriodicTableEmbedder()
        bound1 = ptable.get_group(graph.nodes.species) >= 2
        bound2 = ptable.get_group(graph.nodes.species) <= 11
        transition_metals = (bound1 & bound2).astype(np.float32)
        transition_metals /= transition_metals.sum()
        first_node = jax.random.choice(
            k, np.arange(0, len(graph.nodes.positions)), p=transition_metals
        )
    elif heavy_first and (graph.nodes.species != 0).sum() > 0:
        heavy_indices = np.argwhere(graph.nodes.species != 0).squeeze(-1)
        first_node = jax.random.choice(k, heavy_indices)
    else:
        first_node = jax.random.choice(k, np.arange(0, len(graph.nodes.positions)))
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

    num_nodes = graph.nodes.positions.shape[0]
    target_species_probability = np.zeros((num_nodes, num_species))
    target_species_probability[first_node] = _normalized_bitcount(
        graph.nodes.species[targets], num_species
    )

    rng, k = jax.random.split(rng)
    target_nodes = pick_targets(
        k,
        targets,
        graph.nodes.species,
        target_species_probability[first_node],
        max_targets_per_graph,
    )

    sample = _into_fragment(
        graph,
        visited=np.array([first_node]),
        focus_node=first_node,
        target_species_probability=target_species_probability,
        target_nodes=target_nodes,
        stop=False,
        max_targets_per_graph=max_targets_per_graph,
    )

    rng, k = jax.random.split(rng)
    next_node = jax.random.choice(k, target_nodes)
    visited = np.array([first_node, next_node])
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
    heavy_first,
    max_targets_per_graph,
):
    n_nodes = len(graph.nodes.positions)
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

    counts = np.zeros((n_nodes, num_species))
    for focus_node in range(n_nodes):
        targets = receivers[(senders == focus_node) & mask]
        counts[focus_node] = np.bincount(
            graph.nodes.species[targets], minlength=num_species
        )

    if np.sum(counts) == 0:
        raise ValueError("No targets found.")

    target_species_probability = counts / np.sum(counts)

    # Pick a random focus node
    rng, k = jax.random.split(rng)
    focus_probability = _normalized_bitcount(senders[mask], n_nodes)
    focus_node = jax.random.choice(k, n_nodes, p=focus_probability)
    focus_node = int(focus_node)

    # Pick random targets
    rng, k = jax.random.split(rng)
    targets = receivers[(senders == focus_node) & mask]

    target_nodes = pick_targets(
        k,
        targets,
        graph.nodes.species,
        target_species_probability[focus_node],
        max_targets_per_graph,
    )

    sample = _into_fragment(
        graph=graph,
        visited=visited,
        focus_node=focus_node,
        target_species_probability=target_species_probability,
        target_nodes=target_nodes,
        stop=False,
        max_targets_per_graph=max_targets_per_graph,
    )

    rng, k = jax.random.split(rng)
    next_node = jax.random.choice(k, target_nodes)
    visited = np.concatenate([visited, [next_node]])
    return rng, visited, sample


def _make_last_fragment(graph, num_species, max_targets_per_graph):
    n_nodes = len(graph.nodes.positions)
    return _into_fragment(
        graph,
        visited=np.arange(n_nodes),
        focus_node=0,
        target_species_probability=np.zeros((n_nodes, num_species)),
        target_nodes=np.asarray([0]),
        stop=True,
        max_targets_per_graph=max_targets_per_graph,
    )


def _into_fragment(
    graph,
    visited,
    focus_node,
    target_species_probability,
    target_nodes,
    stop,
    max_targets_per_graph,
):
    pos = graph.nodes.positions
    species = graph.nodes.species

    # Check that all target species are the same.
    target_species = species[target_nodes]
    assert np.all(target_species == target_species[0])

    padded_target_nodes = np.pad(
        target_nodes, (0, max_targets_per_graph - len(target_nodes))
    )
    target_positions_mask = np.zeros(max_targets_per_graph, dtype=bool)
    target_positions_mask[: len(target_nodes)] = not stop
    nodes = datatypes.FragmentsNodes(
        positions=pos,
        species=species,
        focus_and_target_species_probs=target_species_probability,
    )
    globals = datatypes.FragmentsGlobals(
        stop=np.asarray(stop, dtype=bool),
        target_species=target_species[0],
        target_positions=pos[padded_target_nodes] - pos[focus_node],
        target_positions_mask=target_positions_mask,
    )
    globals = jax.tree_map(lambda x: np.expand_dims(x, axis=0), globals)
    graph = graph._replace(nodes=nodes, globals=globals)

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


def _normalized_bitcount(xs: np.ndarray, n: int) -> np.ndarray:
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
