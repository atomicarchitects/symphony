from typing import Iterator, Sequence, Dict

import functools
import ase
import ase.io
import jax
import jax.numpy as jnp
import jraph
import matscipy.neighbours
import numpy as np
import chex
import roundmantissa

import datatypes
import dynamic_batcher
import qm9


def get_datasets(
    rng: chex.PRNGKey,
    root_dir: str,
    nn_tolerance: float,
    nn_cutoff: float,
    max_n_nodes: int,
    max_n_edges: int,
    max_n_graphs: int,
) -> Dict[str, Iterator[datatypes.Fragment]]:
    """Dataloader for the generative model.
    Args:
        rng: The random number seed.
        root_dir: The directory where the QM9 dataset is stored.
        nn_tolerance: The tolerance in Angstroms for the nearest neighbor search. (Maybe 0.1A or 0.5A is good?)
        nn_cutoff: The cutoff in Angstroms for the nearest neighbor search. (Maybe 5A)
        max_n_nodes:
        max_n_edges:
        max_n_graphs:
    Returns:
        An iterator of (batched and padded) fragments.
    """
    # Atomic numbers map to elements H, C, N, O, F.
    atomic_numbers = jnp.array([1, 6, 7, 8, 9])

    all_molecules = qm9.load_qm9(root_dir)
    rng, rng_shuffle = jax.random.split(rng)
    indices = jax.random.permutation(rng_shuffle, len(all_molecules))

    rng, train_rng, val_rng, test_rng = jax.random.split(rng, 4)
    rngs = {
        "train": train_rng,
        "val": val_rng,
        "test": test_rng,
    }
    indices = {
        "train": indices[:50000],
        "val": indices[50000:55000],
        "test": indices[55000:],
    }
    molecules = {
        split: [all_molecules[i] for i in indices[split]]
        for split in ["train", "val", "test"]
    }
    return {
        split: dataloader(
            rngs[split],
            molecules[split],
            atomic_numbers,
            nn_tolerance,
            nn_cutoff,
            max_n_nodes,
            max_n_edges,
            max_n_graphs,
        )
        for split in ["train", "val", "test"]
    }


def dataloader(
    rng: chex.PRNGKey,
    molecules: Sequence[ase.Atoms],
    atomic_numbers: jnp.ndarray,
    nn_tolerance: float,
    nn_cutoff: float,
    max_n_nodes: int,
    max_n_edges: int,
    max_n_graphs: int,
) -> Iterator[datatypes.Fragment]:
    """Dataloader for the generative model.
    Args:
        rng: The random number seed.
        molecules: The molecules to sample from. Each molecule is an ase.Atoms object.
        atomic_numbers: The atomic numbers of the target species. For example, [1, 8] such that [H, O] maps to [0, 1].
        nn_tolerance: The tolerance in Angstroms for the nearest neighbor search. (Maybe 0.1A or 0.5A is good?)
        nn_cutoff: The cutoff in Angstroms for the nearest neighbor search. (Maybe 5A)
        max_n_nodes:
        max_n_edges:
        max_n_graphs:
    Returns:
        An iterator of (batched and padded) fragments.
    """

    graph_molecules = [
        ase_atoms_to_jraph_graph(molecule, atomic_numbers, nn_cutoff)
        for molecule in molecules
    ]
    assert all([isinstance(graph, jraph.GraphsTuple) for graph in graph_molecules])

    for graphs in dynamic_batcher.dynamically_batch(
        fragments_pool_iterator(
            rng, graph_molecules, len(atomic_numbers), nn_tolerance
        ),
        max_n_nodes,
        max_n_edges,
        max_n_graphs,
    ):
        yield pad_graph_to_nearest_ceil_mantissa(
            graphs,
            n_mantissa_bits=1,
            n_max_nodes=max_n_nodes,
            n_max_edges=max_n_edges,
            n_min_graphs=max_n_graphs,
            n_max_graphs=max_n_graphs,
        )


def fragments_pool_iterator(
    rng: chex.PRNGKey,
    graph_molecules: Sequence[jraph.GraphsTuple],
    n_species: int,
    nn_tolerance: float,
) -> Iterator[datatypes.Fragment]:
    """A pool of fragments that are generated on the fly."""
    # TODO: Make this configurable.
    SAMPLES_POOL_SIZE = 1024

    fragments = []
    while True:
        while len(fragments) < SAMPLES_POOL_SIZE:
            rng, index_rng, fragment_rng = jax.random.split(rng, num=3)
            indices = jax.random.randint(index_rng, (), 0, len(graph_molecules))
            fragments += list(
                generate_fragments(
                    fragment_rng, graph_molecules[indices], n_species, nn_tolerance
                )
            )
            assert all([isinstance(sample, jraph.GraphsTuple) for sample in fragments])

        rng, index_rng = jax.random.split(rng)
        index = jax.random.randint(index_rng, (), 0, len(fragments))
        yield fragments.pop(index)


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
        7 nodes --> 8 nodes (2^3)
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

    pad_nodes_to = roundmantissa.ceil_mantissa(n_nodes + 1, n_mantissa_bits)
    pad_edges_to = roundmantissa.ceil_mantissa(n_edges, n_mantissa_bits)
    pad_graphs_to = roundmantissa.ceil_mantissa(n_graphs + 1, n_mantissa_bits)

    pad_nodes_to = np.clip(pad_nodes_to, n_min_nodes, n_max_nodes)
    pad_edges_to = np.clip(pad_edges_to, n_min_edges, n_max_edges)
    pad_graphs_to = np.clip(pad_graphs_to, n_min_graphs, n_max_graphs)

    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )


def ase_atoms_to_jraph_graph(
    atoms: ase.Atoms, atomic_numbers: np.ndarray, nn_cutoff: float
) -> jraph.GraphsTuple:
    # Create edges
    receivers, senders = matscipy.neighbours.neighbour_list(
        quantities="ij", positions=atoms.positions, cutoff=nn_cutoff, cell=np.eye(3)
    )

    # Get the species indices
    species = np.searchsorted(atomic_numbers, atoms.numbers)

    return jraph.GraphsTuple(
        nodes=datatypes.NodesInfo(atoms.positions, species),
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
    nn_tolerance: float = 0.01,
) -> Iterator[datatypes.Fragment]:
    """Generative sequence for a molecular graph.

    Args:
        rng: The random number generator.
        graph: The molecular graph.
        atomic_numbers: The atomic numbers of the target species.
        nn_tolerance: Tolerance for the nearest neighbours.

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
        rng, graph, dist, n_species, nn_tolerance
    )
    yield frag

    for _ in range(n - 2):
        rng, visited_nodes, frag = _make_middle_fragment(
            rng, visited_nodes, graph, dist, n_species, nn_tolerance
        )
        yield frag

    assert len(visited_nodes) == n

    yield _make_last_fragment(graph, n_species)


def _make_first_fragment(rng, graph, dist, n_species, nn_tolerance):
    # pick a random initial node
    rng, k = jax.random.split(rng)
    first_node = jax.random.randint(
        k, shape=(), minval=0, maxval=len(graph.nodes.positions)
    )

    min_dist = dist[graph.senders == first_node].min()
    targets = graph.receivers[
        (graph.senders == first_node) & (dist < min_dist + nn_tolerance)
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
        target_species_probability=species_probability,
        target_node=target,
        stop=False,
    )

    visited = jnp.array([first_node, target])
    return rng, visited, sample


def _make_middle_fragment(rng, visited, graph, dist, n_species, nn_tolerance):
    n_nodes = len(graph.nodes.positions)
    senders, receivers = graph.senders, graph.receivers

    mask = jnp.isin(senders, visited) & ~jnp.isin(receivers, visited)

    min_dist = dist[mask].min()
    mask = mask & (dist < min_dist + nn_tolerance)

    focus_probability = _normalized_bitcount(senders[mask], n_nodes)

    # pick a random focus node
    rng, k = jax.random.split(rng)
    focus_node = jax.random.choice(k, n_nodes, p=focus_probability)

    # target_species_probability
    targets = receivers[(senders == focus_node) & mask]
    target_species_probability = _normalized_bitcount(
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
        target_species_probability,
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
        target_species_probability=jnp.zeros((n_species,)),
        target_node=0,
        stop=True,
    )


def _into_fragment(
    graph,
    visited,
    focus_probability,
    focus_node,
    target_species_probability,
    target_node,
    stop,
):
    nodes = datatypes.FragmentNodes(
        positions=graph.nodes.positions,
        species=graph.nodes.species,
        focus_probability=focus_probability,
    )
    globals = datatypes.FragmentGlobals(
        stop=jnp.array([stop], dtype=bool),  # [1]
        target_species_probability=target_species_probability[None],  # [1, n_species]
        target_species=graph.nodes.species[target_node][None],  # [1]
        target_positions=(
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


@functools.partial(jax.jit, static_argnums=(1,))
def _normalized_bitcount(xs, n: int):
    assert xs.ndim == 1
    return jnp.bincount(xs, length=n) / len(xs)
