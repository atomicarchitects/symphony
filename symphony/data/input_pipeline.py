from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import ase
import ase.io
import chex
import jax
import jax.numpy as jnp
import jraph
import matscipy.neighbours
import ml_collections
import numpy as np
import roundmantissa

from symphony import datatypes
from symphony.data import dynamic_batcher
from symphony.data import fragments as fragments_lib
from symphony.data import qm9


def get_raw_datasets(
    rng: chex.PRNGKey,
    config: ml_collections.ConfigDict,
    root_dir: Optional[str] = None,
) -> Tuple[Dict[str, chex.PRNGKey], jnp.ndarray, Dict[str, List[ase.Atoms]]]:
    """Constructs the splits for the QM9 dataset.
    Args:
        rng: The random number seed.
        config: The configuration.
    Returns:
        An iterator of (batched and padded) fragments.
    """
    # Load all molecules.
    if root_dir is None:
        root_dir = config.root_dir
    all_molecules = qm9.load_qm9(root_dir)

    # Atomic numbers map to elements H, C, N, O, F.
    atomic_numbers = jnp.array([1, 6, 7, 8, 9])

    # Get different randomness for each split.
    rng, train_rng, val_rng, test_rng = jax.random.split(rng, 4)
    rngs = {
        "train": train_rng,
        "val": val_rng,
        "test": test_rng,
    }

    # Construct partitions of the dataset, to create each split.
    # Each partition is a list of indices into all_molecules.
    # TODO is this what we're using?
    # rng, rng_shuffle = jax.random.split(rng)
    # indices = jax.random.permutation(rng_shuffle, len(all_molecules))
    # graphs_cumsum = np.cumsum(
    #     [config.num_train_graphs, config.num_val_graphs, config.num_test_graphs]
    # )
    indices = {
        # "train": indices[: graphs_cumsum[0]],
        # "val": indices[graphs_cumsum[0] : graphs_cumsum[1]],
        # "test": indices[graphs_cumsum[1] : graphs_cumsum[2]],
        "train": range(*config.train_molecules),
        "val": range(*config.val_molecules),
        "test": range(
            config.test_molecules[0], min(config.test_molecules[1], len(all_molecules))
        ),
    }
    molecules = {
        split: [all_molecules[i] for i in indices[split]]
        for split in ["train", "val", "test"]
    }

    return rngs, atomic_numbers, molecules


def get_datasets(
    rng: chex.PRNGKey,
    config: ml_collections.ConfigDict,
) -> Dict[str, Iterator[datatypes.Fragments]]:
    """Dataloader for the generative model for each split.
    Args:
        rng: The random number seed.
        config: The configuration.
    Returns:
        An iterator of (batched and padded) fragments.
    """
    rngs, atomic_numbers, molecules = get_raw_datasets(rng, config)
    return {
        split: dataloader(
            rngs[split],
            molecules[split],
            atomic_numbers,
            config.nn_tolerance,
            config.nn_cutoff,
            config.max_n_nodes,
            config.max_n_edges,
            config.max_n_graphs,
            max_iterations=10 if split in ["val", "test"] else None,
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
    max_iterations: Optional[int] = None,
) -> Iterator[datatypes.Fragments]:
    """Dataloader for the generative model.
    Args:
        rng: The random number seed.
        molecules: The molecules to sample from. Each molecule is an ase.Atoms object.
        atomic_numbers: The atomic numbers of the target species. For example, [1, 8] such that [H, O] maps to [0, 1].
        nn_tolerance: The tolerance in Angstroms for the nearest neighbor search. Only atoms upto (min_nn_dist + nn_tolerance) distance away will be considered as neighbors to the current atom. (Maybe 0.1A or 0.5A is good?)
        nn_cutoff: The cutoff in Angstroms for the nearest neighbor search. Only atoms upto cutoff distance away will be considered as neighbors to the current atom. (Maybe 5A)
        max_n_nodes: The maximum number of nodes in a batch after padding.
        max_n_edges: The maximum number of nodes in a batch after padding.
        max_n_graphs: The maximum number of nodes in a batch after padding.
    Returns:
        An iterator of (batched and padded) fragments.
    """

    graph_molecules = [
        ase_atoms_to_jraph_graph(molecule, atomic_numbers, nn_cutoff)
        for molecule in molecules
    ]
    assert all([isinstance(graph, jraph.GraphsTuple) for graph in graph_molecules])

    for iteration, graphs in enumerate(
        dynamic_batcher.dynamically_batch(
            fragments_pool_iterator(
                rng, graph_molecules, len(atomic_numbers), nn_tolerance
            ),
            max_n_nodes,
            max_n_edges,
            max_n_graphs,
        )
    ):
        if max_iterations is not None and iteration == max_iterations:
            break

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
) -> Iterator[datatypes.Fragments]:
    """A pool of fragments that are generated on the fly."""
    # TODO: Make this configurable.
    SAMPLES_POOL_SIZE = 1024

    fragments = []
    while True:
        while len(fragments) < SAMPLES_POOL_SIZE:
            rng, index_rng, fragment_rng = jax.random.split(rng, num=3)
            indices = jax.random.randint(index_rng, (), 0, len(graph_molecules))
            fragments += list(
                fragments_lib.generate_fragments(
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
    """Pads a batched graph to a rounded number of nodes, edges, and graphs.

    The roundind is done in the mantissa, see `roundmantissa.ceil_mantissa`.
    After rounding, the number of nodes, edges, and graphs is clipped to the
    specified min and max values.

    Args:
        graphs_tuple: a batched `jraph.GraphsTuple`

    Returns:
        A padded `jraph.GraphsTuple`.
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
    atoms: ase.Atoms, atomic_numbers: jnp.ndarray, nn_cutoff: float, cell: np.ndarray = np.eye(3)
) -> jraph.GraphsTuple:
    # Create edges
    receivers, senders = matscipy.neighbours.neighbour_list(
<<<<<<< HEAD
        quantities="ij", positions=atoms.positions, cutoff=nn_cutoff, cell=cell
=======
        quantities="ij", positions=atoms.positions, cutoff=nn_cutoff,
        cell=np.eye(3),
>>>>>>> 360cf8c4cff0a3a3532c9f79c216c128e309b776
    )

    # Get the species indices
    species = np.searchsorted(atomic_numbers, atoms.numbers)

    return jraph.GraphsTuple(
        nodes=datatypes.NodesInfo(np.asarray(atoms.positions), np.asarray(species)),
        edges=np.ones(len(senders)),
        globals=None,
        senders=np.asarray(senders),
        receivers=np.asarray(receivers),
        n_node=np.array([len(atoms)]),
        n_edge=np.array([len(senders)]),
    )
