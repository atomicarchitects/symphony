import argparse
import logging
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from input_pipeline import ase_atoms_to_jraph_graph, generate_fragments
from qm9 import load_qm9


def main(seed: int = 0, start: int = 0, end: int = 3000, output: str = "fragments.pkl"):
    seed = jax.random.PRNGKey(seed)
    molecules = load_qm9("qm9_data")[start:end]

    atomic_numbers = jnp.array([1, 6, 7, 8, 9])
    epsilon = 0.125  # Angstroms
    cutoff = 5.0  # Angstroms
    filtering_threshold = 2.0  # Angstroms

    fragments = []

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

        fragments.extend(frags)

    with open(output, "wb") as f:
        pickle.dump(fragments, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=3000)
    parser.add_argument("--output", type=str, default="fragments.pkl")
    args = parser.parse_args()
    main(**vars(args))
