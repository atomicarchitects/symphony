import argparse
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm

from input_pipeline import ase_atoms_to_jraph_graph, generate_fragments
from qm9 import load_qm9


def main(seed: int = 0, start: int = 0, end: int = 3000, output: str = "fragments.pkl"):
    seed = jax.random.PRNGKey(seed)
    molecules = load_qm9("qm9_data")

    if start is not None and end is not None:
        molecules = molecules[start:end]

    atomic_numbers = jnp.array([1, 6, 7, 8, 9])
    epsilon = 0.125  # Angstroms
    cutoff = 5.0  # Angstroms
    filtering_threshold = 2.0  # Angstroms

    signature = {
        # nodes
        "positions": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        "species": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "focus_probability": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        # edges
        "senders": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "receivers": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        # globals
        "stop": tf.TensorSpec(shape=(1,), dtype=tf.bool),
        "target_positions": tf.TensorSpec(shape=(1, 3), dtype=tf.float32),
        "target_species": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "target_species_probability": tf.TensorSpec(shape=(1, len(atomic_numbers)), dtype=tf.float32),
        # n_node and n_edge
        "n_node": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "n_edge": tf.TensorSpec(shape=(1,), dtype=tf.int32),
    }

    def generator():
        for molecule in tqdm.tqdm(molecules):
            graph = ase_atoms_to_jraph_graph(molecule, atomic_numbers, cutoff)
            frags = generate_fragments(seed, graph, len(atomic_numbers), epsilon)
            frags = list(frags)

            skip = False
            for frag in frags:
                d = np.linalg.norm(frag.globals.target_positions)
                if d > filtering_threshold:
                    print("Target position is too far away from the rest of the molecule.")
                    skip = True

            if skip:
                continue

            for frag in frags:
                yield {
                    "positions": frag.nodes.positions.astype(np.float32),
                    "species": frag.nodes.species.astype(np.int32),
                    "focus_probability": frag.nodes.focus_probability.astype(np.float32),
                    "senders": frag.senders.astype(np.int32),
                    "receivers": frag.receivers.astype(np.int32),
                    "stop": frag.globals.stop.astype(np.bool_),
                    "target_positions": frag.globals.target_positions.astype(np.float32),
                    "target_species": frag.globals.target_species.astype(np.int32),
                    "target_species_probability": frag.globals.target_species_probability.astype(np.float32),
                    "n_node": frag.n_node.astype(np.int32),
                    "n_edge": frag.n_edge.astype(np.int32),
                }

    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)

    os.mkdir(output)
    dataset.save(output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output", type=str, default="fragments")
    args = parser.parse_args()
    main(**vars(args))
