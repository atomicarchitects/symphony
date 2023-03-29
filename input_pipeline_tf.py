"""Input pipeline for the QM9 dataset with the tf.data API."""

import functools
from typing import Dict, List, Sequence, Tuple
import re
import os

from absl import logging
import tensorflow as tf
import chex
import jax
import numpy as np
import jraph
import ml_collections

import datatypes


def get_datasets(
    rng: chex.PRNGKey, config: ml_collections.ConfigDict
) -> Dict[str, tf.data.Dataset]:
    """Loads and preprocesses the QM9 dataset as tf.data.Datasets for each split."""
    del rng

    # Get the raw datasets.
    datasets = get_raw_qm9_datasets(
        config.root_dir,
        config.train_molecules,
        config.val_molecules,
        config.test_molecules,
    )

    # Convert to jraph.GraphsTuple.
    for split, dataset_split in datasets.items():
        datasets[split] = dataset_split.map(
            _convert_to_graphstuple,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

    # Estimate the padding budget.
    if config.compute_padding_dynamically:
        max_n_nodes, max_n_edges, max_n_graphs = estimate_padding_budget_for_num_graphs(
            datasets["train"], config.max_n_graphs, num_estimation_graphs=1000
        )

    else:
        max_n_nodes, max_n_edges, max_n_graphs = (
            config.max_n_nodes,
            config.max_n_edges,
            config.max_n_graphs,
        )

    logging.info(
        "Padding budget %s as: n_nodes = %d, n_edges = %d, n_graphs = %d",
        "computed" if config.compute_padding_dynamically else "provided",
        max_n_nodes,
        max_n_edges,
        max_n_graphs,
    )

    # Pad an example graph to see what the output shapes will be.
    # We will use this shape information when creating the tf.data.Dataset.
    example_graph = next(datasets["train"].as_numpy_iterator())
    example_padded_graph = jraph.pad_with_graphs(
        example_graph, n_node=max_n_nodes, n_edge=max_n_edges, n_graph=max_n_graphs
    )
    padded_graphs_spec = _specs_from_graphs_tuple(example_padded_graph)

    # Batch and pad each split separately.
    for split, dataset_split in datasets.items():
        batching_fn = functools.partial(
            jraph.dynamically_batch,
            graphs_tuple_iterator=iter(dataset_split),
            n_node=max_n_nodes,
            n_edge=max_n_edges,
            n_graph=max_n_graphs,
        )
        datasets[split] = tf.data.Dataset.from_generator(
            batching_fn, output_signature=padded_graphs_spec
        )

    return datasets


def estimate_padding_budget_for_num_graphs(
    dataset: tf.data.Dataset, num_graphs: int, num_estimation_graphs: int
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
    for graph in dataset.take(num_estimation_graphs).as_numpy_iterator():
        n_node, n_edge, n_graph = get_graphs_tuple_size(graph)
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


def _deprecated_get_raw_qm9_datasets(
    rng: chex.PRNGKey,
    root_dir: str,
    num_train_files: int,
    num_val_files: int,
    num_test_files: int,
) -> Dict[str, tf.data.Dataset]:
    """Loads the raw QM9 dataset as tf.data.Datasets for each split."""
    print(root_dir)
    # Root directory of the dataset.
    filenames = os.listdir(root_dir)
    filenames = [os.path.join(root_dir, f) for f in filenames if "dataset_tf" in f]

    # Shuffle the filenames.
    shuffled_indices = jax.random.permutation(rng, len(filenames))
    shuffled_filenames = [filenames[i] for i in shuffled_indices]

    # Partition the filenames into train, val, and test.
    num_files_cumsum = np.cumsum([num_train_files, num_val_files, num_test_files])
    files_by_split = {
        "train": shuffled_filenames[: num_files_cumsum[0]],
        "val": shuffled_filenames[num_files_cumsum[0] : num_files_cumsum[1]],
        "test": shuffled_filenames[num_files_cumsum[1] : num_files_cumsum[2]],
    }

    element_spec = tf.data.Dataset.load(filenames[0]).element_spec
    datasets = {}
    for split, files_split in files_by_split.items():
        dataset_split = tf.data.Dataset.from_tensor_slices(files_split)
        dataset_split = dataset_split.interleave(
            lambda x: tf.data.Dataset.load(x, element_spec=element_spec),
            cycle_length=4,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        datasets[split] = dataset_split
    return datasets


def get_raw_qm9_datasets(
    root_dir: str,
    train_molecules: Tuple[int, int],
    val_molecules: Tuple[int, int],
    test_molecules: Tuple[int, int],
    seed: int = 0,
) -> Dict[str, tf.data.Dataset]:
    """Loads the raw QM9 dataset as tf.data.Datasets for each split."""
    # Root directory of the dataset.
    filenames = os.listdir(root_dir)
    filenames = [
        os.path.join(root_dir, f) for f in filenames if f.startswith("fragments_seed")
    ]

    # Partition the filenames into train, val, and test.
    def filter_by_molecule_number(
        filenames: Sequence[str], start: int, end: int
    ) -> List[str]:
        def filter_file(filename: str, start: int, end: int) -> bool:
            filename = os.path.basename(filename)
            _, file_start, file_end = [int(val) for val in re.findall(r"\d+", filename)]
            return start <= file_start and file_end <= end

        return [f for f in filenames if filter_file(f, start, end)]

    files_by_split = {
        "train": filter_by_molecule_number(filenames, *train_molecules),
        "val": filter_by_molecule_number(filenames, *val_molecules),
        "test": filter_by_molecule_number(filenames, *test_molecules),
    }

    element_spec = tf.data.Dataset.load(filenames[0]).element_spec
    datasets = {}
    for split, files_split in files_by_split.items():
        dataset_split = tf.data.Dataset.from_tensor_slices(files_split)
        dataset_split = dataset_split.interleave(
            lambda x: tf.data.Dataset.load(x, element_spec=element_spec),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        dataset_split = dataset_split.shuffle(1000, seed=seed)
        datasets[split] = dataset_split
    return datasets


def _specs_from_graphs_tuple(graph: jraph.GraphsTuple):
    """Returns a tf.TensorSpec corresponding to this graph."""

    def get_tensor_spec(array: np.ndarray):
        shape = list(array.shape)
        dtype = array.dtype
        return tf.TensorSpec(shape=shape, dtype=dtype)

    return jraph.GraphsTuple(
        nodes=datatypes.FragmentsNodes(
            positions=get_tensor_spec(graph.nodes.positions),
            species=get_tensor_spec(graph.nodes.species),
            focus_probability=get_tensor_spec(graph.nodes.focus_probability),
        ),
        globals=datatypes.FragmentsGlobals(
            stop=get_tensor_spec(graph.globals.stop),
            target_positions=get_tensor_spec(graph.globals.target_positions),
            target_species=get_tensor_spec(graph.globals.target_species),
            target_species_probability=get_tensor_spec(
                graph.globals.target_species_probability
            ),
        ),
        edges=get_tensor_spec(graph.edges),
        receivers=get_tensor_spec(graph.receivers),
        senders=get_tensor_spec(graph.senders),
        n_node=get_tensor_spec(graph.n_node),
        n_edge=get_tensor_spec(graph.n_edge),
    )


def _convert_to_graphstuple(graph: Dict[str, tf.Tensor]) -> jraph.GraphsTuple:
    """Converts a dictionary of tf.Tensors to a GraphsTuple."""
    positions = graph["positions"]
    species = graph["species"]
    focus_probability = graph["focus_probability"]
    receivers = graph["receivers"]
    senders = graph["senders"]
    n_node = graph["n_node"]
    n_edge = graph["n_edge"]
    edges = tf.ones((tf.shape(senders)[0], 1))
    stop = graph["stop"]
    target_positions = graph["target_positions"]
    target_species = graph["target_species"]
    target_species_probability = graph["target_species_probability"]

    return jraph.GraphsTuple(
        nodes=datatypes.FragmentsNodes(
            positions=positions, species=species, focus_probability=focus_probability
        ),
        edges=edges,
        receivers=receivers,
        senders=senders,
        globals=datatypes.FragmentsGlobals(
            stop=stop,
            target_positions=target_positions,
            target_species=target_species,
            target_species_probability=target_species_probability,
        ),
        n_node=n_node,
        n_edge=n_edge,
    )
