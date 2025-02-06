from typing import Dict, Iterator, Optional, Sequence, Tuple
import functools

from absl import logging
import ase
import ase.io
import chex
import jax
import jax.numpy as jnp
import jraph
import matscipy.neighbours
import ml_collections
import numpy as np

from symphony import datatypes
from symphony.data import fragments, datasets


def infer_edges_with_radial_cutoff_on_positions(
    structure: datatypes.Structures, radial_cutoff: float
) -> datatypes.Structures:
    """Infer edges from node positions, using a radial cutoff."""
    assert structure.n_node.shape[0] == 1, "Only one structure is supported."

    receivers, senders = matscipy.neighbours.neighbour_list(
        quantities="ij",
        positions=structure.nodes.positions,
        cutoff=radial_cutoff,
        cell=np.eye(3),
    )

    return structure._replace(
        edges=np.ones(len(senders)),
        senders=np.asarray(senders),
        receivers=np.asarray(receivers),
        n_edge=np.array([len(senders)]),
        globals=datatypes.GlobalsInfo(
            num_residues=structure.globals.num_residues,
            residue_starts=structure.globals.residue_starts,
            n_short_edge=jnp.array([len(senders)]),
            n_long_edge=None,
        ),
    )


def get_random_edges(
    rng: chex.PRNGKey, structure: datatypes.Structures, num_edges: int
):
    """Get random edges for a structure."""
    n_node = structure.n_node[0]
    n_edge = structure.n_edge[0]
    assert num_edges >= n_edge, "Number of edges must be greater than the current number of edges."
    total_n_edge = n_node * (n_node - 1) // 2
    senders = jnp.repeat(jnp.arange(n_node), n_node)
    receivers = jnp.tile(jnp.arange(n_node), n_node)
    num_edges = min(total_n_edge, num_edges)
    indices = jax.random.choice(rng, total_n_edge, (num_edges - n_edge,), replace=False)
    # indices = jnp.pad(indices, (0, total_n_edge - num_edges))
    return structure._replace(
        edges=np.ones(num_edges),
        senders=jnp.concatenate([structure.senders, senders[indices]]),
        receivers=jnp.concatenate([structure.receivers, receivers[indices]]),
        n_edge=np.array([num_edges]),
        globals=datatypes.GlobalsInfo(
            num_residues=structure.globals.num_residues,
            residue_starts=structure.globals.residue_starts,
            n_short_edge=structure.globals.n_short_edge,
            n_long_edge=jnp.array([num_edges - n_edge]),
        ),
    )


def create_fragments_dataset(
    rng: chex.PRNGKey,
    structures: Sequence[datatypes.Structures],
    keep_indices: Sequence[int],
    num_species: int,
    infer_edges_with_radial_cutoff: bool,
    radial_cutoff: float,
    use_same_rng_across_structures: bool,
    fragment_logic: str,
    heavy_first: bool,
    max_targets_per_graph: int,
    num_seeds: int,
    max_edges: Optional[int] = None,
    max_num_residues: Optional[int] = None,
    max_radius: Optional[float] = None,
    nn_tolerance: Optional[float] = None,
    transition_first: Optional[bool] = False,
    n_terminus: Optional[bool] = False,
    fragment_number: Optional[int] = -1,
) -> Iterator[datatypes.Fragments]:
    """Creates an iterator of fragments from a sequence of structures."""
    if infer_edges_with_radial_cutoff and radial_cutoff is None:
        raise ValueError("radial_cutoff must be provided if infer_edges is True.")
    if not infer_edges_with_radial_cutoff and max_edges is None:
        raise ValueError("max_edges must be provided if infer_edges is False.")

    def fragment_generator(rng: chex.PRNGKey):
        """Generates fragments for a split."""
        # Loop indefinitely.
        while True:
            for seed in range(num_seeds):
                seed_rng = jax.random.fold_in(rng, seed)
                for index in keep_indices:
                    structure = structures[index]
                    if use_same_rng_across_structures:
                        structure_rng = seed_rng
                    else:
                        structure_rng = jax.random.fold_in(seed_rng, index)
                    
                    if max_num_residues and structure.globals.num_residues > max_num_residues:
                        _, ndx_rng = jax.random.split(structure_rng)
                        start_residue = jax.random.randint(ndx_rng, (1,), 0, structure.globals.num_residues - max_num_residues)[0]
                        end_residue = start_residue + max_num_residues
                        start_ndx = structure.globals.residue_starts[start_residue]
                        end_ndx = structure.globals.residue_starts[end_residue]
                        structure = structure._replace(
                            nodes=datatypes.NodesInfo(
                                structure.nodes.positions[start_ndx:end_ndx],
                                structure.nodes.species[start_ndx:end_ndx],
                            ),
                            n_node=np.array([end_ndx - start_ndx]),
                        )

                    # if infer_edges_with_radial_cutoff:
                    #     if structure.n_edge is not None:
                    #         raise ValueError("Structure already has edges.")
                    #     structure = infer_edges_with_radial_cutoff_on_positions(
                    #         structure, radial_cutoff=radial_cutoff
                    #     )
                    
                    # else:
                    #     if structure.n_edge is None:
                    #         structure = get_random_edges(
                    #             structure_rng, structure, num_edges=max_edges
                    #         )

                    if structure.n_edge is not None:
                        raise ValueError("Structure already has edges.")
                    structure = infer_edges_with_radial_cutoff_on_positions(
                        structure, radial_cutoff=radial_cutoff
                    )
                    structure = get_random_edges(
                        structure_rng, structure, num_edges=max_edges
                    )

                    frag_generator = fragments.generate_fragments(
                        rng=structure_rng,
                        graph=structure,
                        num_species=num_species,
                        nn_tolerance=nn_tolerance,
                        max_radius=max_radius,
                        mode=fragment_logic,
                        max_targets_per_graph=max_targets_per_graph,
                        heavy_first=heavy_first,
                        transition_first=transition_first,
                        n_terminus=n_terminus,
                    )

                    if fragment_number == -1:
                        yield from frag_generator
                    else:
                        for i, frag in enumerate(frag_generator):
                            if i == fragment_number:
                                yield frag
                                break

    return fragment_generator(rng)


def estimate_padding_budget(
    fragments_iterator: Iterator[datatypes.Fragments],
    num_graphs: int,
    num_estimation_graphs: int,
) -> Tuple[int, int, int]:
    """Estimates the padding budget for a dataset of unbatched GraphsTuples.
    Args:
        dataset: A dataset of unbatched GraphsTuples.
        num_graphs: The intended number of graphs per batch. Note that no batching is performed by
        this function.
        num_estimation_graphs: How many graphs to take from the dataset to estimate
        the distribution of number of nodes and edges per graph.
    Returns:
        padding_budget: The padding budget for batching and padding the graphs
        in this dataset to the given batch size.
    """

    def get_graphs_tuple_size(graph: datatypes.Fragments) -> Tuple[int, int, int]:
        """Returns the number of nodes, edges and graphs in a GraphsTuple."""
        return (
            np.shape(jax.tree_leaves(graph.nodes)[0])[0],
            np.sum(graph.n_edge),
            np.shape(graph.n_node)[0],
        )

    def next_multiple_of_64(val: float) -> int:
        """Returns the next multiple of 64 after val."""
        return 64 * (1 + int(val // 64))

    if num_graphs <= 1:
        raise ValueError("Batch size must be > 1 to account for padding graphs.")

    total_num_nodes = 0
    total_num_edges = 0
    for step, fragment in enumerate(fragments_iterator):
        if step >= num_estimation_graphs:
            break

        n_node, n_edge, n_graph = get_graphs_tuple_size(fragment)
        if n_graph != 1:
            raise ValueError("Dataset contains batched GraphTuples.")

        total_num_nodes += n_node
        total_num_edges += n_edge

    num_nodes_per_graph_estimate = total_num_nodes / num_estimation_graphs
    num_edges_per_graph_estimate = total_num_edges / num_estimation_graphs

    n_node = next_multiple_of_64(num_nodes_per_graph_estimate * num_graphs)
    n_edge = next_multiple_of_64(num_edges_per_graph_estimate * num_graphs)
    n_graph = num_graphs
    return n_node, n_edge, n_graph


def pad_and_batch_fragments(
    fragments_iterator: Iterator[datatypes.Fragments],
    max_n_nodes: int,
    max_n_edges: int,
    max_n_graphs: int,
    compute_padding_dynamically: bool,
) -> Iterator[datatypes.Fragments]:
    """Pad and batch an iterator of fragments."""

    # Estimate the padding budget.
    if compute_padding_dynamically:
        max_n_nodes, max_n_edges, max_n_graphs = estimate_padding_budget(
            fragments_iterator, max_n_graphs, num_estimation_graphs=1000
        )

    logging.info(
        "Padding budget %s as: n_nodes = %d, n_edges = %d, n_graphs = %d",
        "computed" if compute_padding_dynamically else "provided",
        max_n_nodes,
        max_n_edges,
        max_n_graphs,
    )

    # Now we batch and pad the graphs.
    return jraph.dynamically_batch(
        graphs_tuple_iterator=fragments_iterator,
        n_node=max_n_nodes,
        n_edge=max_n_edges,
        n_graph=max_n_graphs,
    )


def get_datasets(
    rng: chex.PRNGKey,
    config: ml_collections.ConfigDict,
) -> Dict[str, Iterator[datatypes.Fragments]]:
    """Creates the datasets of fragments, as specified by the config."""

    # Get the dataset of structures.
    dataset = datasets.utils.get_dataset(config)
    structures = dataset.structures()
    split_indices = dataset.split_indices()

    # Create the fragments datasets.
    fragments_iterators = {
        split: create_fragments_dataset(
            rng=rng,
            structures=structures,
            keep_indices=split_indices[split],
            num_species=dataset.num_species(),
            infer_edges_with_radial_cutoff=config.infer_edges_with_radial_cutoff,
            radial_cutoff=config.radial_cutoff,
            use_same_rng_across_structures=config.use_same_rng_across_structures,
            fragment_logic=config.fragment_logic,
            nn_tolerance=config.get("nn_tolerance", None),
            max_radius=config.get("max_radius", None),
            num_seeds=config.get("num_frag_seeds", 1),
            heavy_first=config.heavy_first,
            max_edges=config.get("max_edges_per_mol", None),
            max_targets_per_graph=config.max_targets_per_graph,
            max_num_residues=config.get("max_num_residues", None),
            transition_first=config.transition_first,
            fragment_number=config.get("fragment_number", -1),
            n_terminus=config.dataset=="cath" or config.dataset == "miniprotein",
        )
        for split in ["train", "val", "test"]
    }

    # Pad and batch each of the fragments datasets.
    pad_and_batch_fragments_fn = functools.partial(
        pad_and_batch_fragments,
        max_n_nodes=config.max_n_nodes,
        max_n_edges=config.max_n_edges,
        max_n_graphs=config.max_n_graphs,
        compute_padding_dynamically=config.compute_padding_dynamically,
    )
    return {
        split: pad_and_batch_fragments_fn(fragments_iterator)
        for split, fragments_iterator in fragments_iterators.items()
    }


def to_jraph_graph(
    positions: np.ndarray, atom_symbols: np.ndarray, atoms_to_species: Dict[int, int], radial_cutoff: float
) -> jraph.GraphsTuple:
    # Create edges
    receivers, senders = matscipy.neighbours.neighbour_list(
        quantities="ij", positions=positions, cutoff=radial_cutoff, cell=np.eye(3)
    )

    # Get the species indices
    species = np.vectorize(atoms_to_species.get)(atom_symbols)

    return jraph.GraphsTuple(
        nodes=datatypes.NodesInfo(jnp.asarray(positions), jnp.asarray(species)),
        edges=jnp.ones(len(senders)),
        globals=None,
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
        n_node=jnp.array([len(atom_symbols)]),
        n_edge=jnp.array([len(senders)]),
    )
