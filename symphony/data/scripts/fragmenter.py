import argparse
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
import sys

sys.path.append("..")

import input_pipeline  # noqa: E402
import qm9  # noqa: E402


def main(
    seed: int,
    start: int,
    end: int,
    output: str = "fragments.pkl",
    mode: str = "nn",
):
    seed = jax.random.PRNGKey(seed)
    molecules = qm9.load_qm9("qm9_data")

    if start is not None and end is not None:
        molecules = molecules[start:end]

    atomic_numbers = jnp.array([1, 6, 7, 8, 9])
    nn_tolerance = 0.125  # Angstroms
    cutoff = 5.0  # Angstroms
    max_radius = 2.03  # Angstroms

    signature = {
        # nodes
        "positions": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        "species": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "target_species_probs": tf.TensorSpec(
            shape=(None, len(atomic_numbers)), dtype=tf.float32
        ),
        # edges
        "senders": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "receivers": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        # globals
        "stop": tf.TensorSpec(shape=(1,), dtype=tf.bool),
        "target_positions": tf.TensorSpec(shape=(1, 3), dtype=tf.float32),
        "target_species": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        # n_node and n_edge
        "n_node": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "n_edge": tf.TensorSpec(shape=(1,), dtype=tf.int32),
    }

    def generator():
        for molecule in tqdm.tqdm(molecules):
            graph = input_pipeline.ase_atoms_to_jraph_graph(
                molecule, atomic_numbers, cutoff
            )
            frags = input_pipeline.generate_fragments(
                seed, graph, len(atomic_numbers), nn_tolerance, max_radius, mode
            )
            frags = list(frags)

            skip = False
            for frag in frags:
                d = np.linalg.norm(frag.globals.target_positions)
                if d > max_radius:
                    print(
                        "Target position is too far away from the rest of the molecule."
                    )
                    skip = True
            if len(frags) == 0 or not frags[-1].globals.stop:
                print("The last fragment is not a stop fragment.")
                skip = True

            if skip:
                continue

            for frag in frags:
                yield {
                    "positions": frag.nodes.positions.astype(np.float32),
                    "species": frag.nodes.species.astype(np.int32),
                    "target_species_probs": frag.nodes.target_species_probs.astype(
                        np.float32
                    ),
                    "senders": frag.senders.astype(np.int32),
                    "receivers": frag.receivers.astype(np.int32),
                    "stop": frag.globals.stop.astype(np.bool_),
                    "target_positions": frag.globals.target_positions.astype(
                        np.float32
                    ),
                    "target_species": frag.globals.target_species.astype(np.int32),
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
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--output", type=str, default="fragments")
    parser.add_argument("--mode", type=str, default="nn")
    args = parser.parse_args()
    main(**vars(args))
