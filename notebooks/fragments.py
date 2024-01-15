import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import tensorflow as tf
import tqdm
from typing import List, Sequence
sys.path.append("../")

from symphony.data import input_pipeline_tf


workdirs = {
    "nn_1": "/home/songk/potato/songk/silica_fragments/nn/max_targets_1/",
    "nn_4": "/home/songk/potato/songk/silica_fragments/nn/max_targets_4/",
    "radius_1": "/home/songk/potato/songk/silica_fragments/radius/max_targets_1/",
    "radius_4": "/home/songk/potato/songk/silica_fragments/radius/max_targets_4/",
}

def get_dataset(method, seed=0):
    # Set the seed for reproducibility.
    tf.random.set_seed(seed)

    # Root directory of the dataset.
    filenames = sorted(os.listdir(workdirs[method]))
    filenames = [
        os.path.join(workdirs[method], f)
        for f in filenames
        if f.startswith("fragments_")
    ]
    if len(filenames) == 0:
        raise ValueError(f"No files found in {workdirs[method]}.")

    def filter_by_molecule_number(
        filenames: Sequence[str], start: int, end: int
    ) -> List[str]:
        def filter_file(filename: str, start: int, end: int) -> bool:
            filename = os.path.basename(filename)
            _, file_start, file_end = [int(val) for val in re.findall(r"\d+", filename)]
            return start <= file_start and file_end <= end

        return [f for f in filenames if filter_file(f, start, end)]

    # Number of molecules for training can be smaller than the chunk size.
    all_files = filter_by_molecule_number(filenames, 0, 135000),

    element_spec = tf.data.Dataset.load(filenames[0]).element_spec
    dataset = tf.data.Dataset.from_tensor_slices(all_files)
    dataset = dataset.interleave(
        lambda x: tf.data.Dataset.load(x, element_spec=element_spec),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Shuffle the dataset.
    dataset = dataset.shuffle(1000, seed=seed)

    # Convert to jraph.GraphsTuple.
    dataset = dataset.map(
        input_pipeline_tf._convert_to_graphstuple,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    return dataset


num_targets = {"nn_1": [], "nn_4": [], "radius_1": [], "radius_4": []}
for frag_method in workdirs:
    dataset = get_dataset(frag_method)
    for graph in tqdm.tqdm(dataset.as_numpy_iterator()):
        targets = graph.globals.target_position_mask.sum()
        num_targets[frag_method].append(targets)
import pickle
with open("valid_target_counts.pkl", 'wb') as f:
    pickle.dump(num_targets, f)
