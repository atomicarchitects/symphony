from typing import List, Tuple

import logging
import os
from absl import logging
from absl import flags
from absl import app
import tqdm.contrib.concurrent
import ase
import jax
from ml_collections import config_flags
import numpy as np
import pickle
from pymatgen.core.structure import Structure
import tensorflow as tf
import tqdm

from symphony.data import fragments
from symphony.data import input_pipeline, matproj

import configs.silica.default as default

FLAGS = flags.FLAGS


def generate_all_fragments(
    molecules: List[ase.Atoms],
    seed: int,
    start: int,
    end: int,
    output_dir: str,
    cutoffs: Tuple[float],
    min_n_nodes: float,
):
    logging.info(f"Generating fragments {start}:{end} using seed {seed}")
    logging.info(f"Saving to {output_dir}")

    rng = jax.random.PRNGKey(seed)

    if start is not None and end is not None:
        molecules = molecules[start:end]

    # make supercell if structure is too small
    for i in range(len(molecules)):
        num_atoms = molecules[i].numbers.shape[0]
        if num_atoms < min_n_nodes:
            if num_atoms >= min_n_nodes / 2:
                P = np.eye(3)
                p_seed, rng = jax.random.split(rng)
                j = jax.random.choice(p_seed, 3)
                P[j, j] = 2
            elif num_atoms >= min_n_nodes / 4:
                P = 2 * np.eye(3)
                p_seed, rng = jax.random.split(rng)
                j = jax.random.choice(p_seed, 3)
                P[j, j] = 1
            else:
                P = 2 * np.eye(3)
            molecules[i] = ase.build.make_supercell(molecules[i], P)

    atomic_numbers = np.array([8, 14])

    signature = {
        # nodes
        "positions": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        "species": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "focus_and_target_species_probs": tf.TensorSpec(
            shape=(None, len(atomic_numbers)), dtype=tf.float32
        ),
        # edges
        "senders": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "receivers": tf.TensorSpec(shape=(None,), dtype=tf.int32),
        "relative_positions": tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
        # globals
        "stop": tf.TensorSpec(shape=(1,), dtype=tf.bool),
        "target_positions": tf.TensorSpec(shape=(1, 3), dtype=tf.float32),
        "target_species": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "cell": tf.TensorSpec(shape=(1, 3, 3), dtype=tf.int32),
        # n_node and n_edge
        "n_node": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "n_edge": tf.TensorSpec(shape=(1,), dtype=tf.int32),
    }

    mol_indices = []  # mol ndx matching up with each fragment

    def generator():
        for mol_ndx, mol in tqdm.tqdm(enumerate(molecules)):
            graph = input_pipeline.ase_atoms_to_jraph_graph(
                mol, atomic_numbers, cutoffs=cutoffs, periodic=True
            )
            frags = fragments.generate_silica_fragments(
                rng,
                graph,
                atomic_numbers,
                FLAGS.nn_tolerance,
                FLAGS.max_radius,
                "nn",
                heavy_first=FLAGS.config.heavy_first
            )
            frags = list(frags)

            mol_indices.extend([mol_ndx] * len(frags))

            for frag in frags:
                yield {
                    "positions": frag.nodes.positions.astype(np.float32),
                    "species": frag.nodes.species.astype(np.int32),
                    "focus_and_target_species_probs": frag.nodes.focus_and_target_species_probs.astype(
                        np.float32
                    ),
                    "senders": frag.senders.astype(np.int32),
                    "receivers": frag.receivers.astype(np.int32),
                    "relative_positions": frag.edges.relative_positions.astype(np.float32),
                    "stop": frag.globals.stop.astype(np.bool_),
                    "target_positions": frag.globals.target_positions.astype(
                        np.float32
                    ),
                    "target_species": frag.globals.target_species.astype(np.int32),
                    "cell": np.expand_dims(frag.globals.cell, 0).astype(np.float32),
                    "n_node": frag.n_node.astype(np.int32),
                    "n_edge": frag.n_edge.astype(np.int32),
                }

    dataset = tf.data.Dataset.from_generator(generator, output_signature=signature)

    os.makedirs(output_dir, exist_ok=True)
    dataset.save(output_dir)
    chunk_start, chunk_end = output_dir.split('/')[-1].split('_')[-2:]
    with open(os.path.join(FLAGS.config.root_dir, f"mol_indices_{chunk_start}_{chunk_end}.pkl"), "wb") as f:
        pickle.dump(mol_indices, f)


def _generate_all_fragments_wrapper(args):
    """Dummy wrapper to allow parallelization."""
    return generate_all_fragments(*args)


def main(unused_argv) -> None:
    del unused_argv

    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)

    # Create a list of arguments to pass to generate_all_fragments
    structures = matproj.get_materials(FLAGS.config.matgen_query)
    molecules = [
        ase.Atoms(
            positions=mol.structure.cart_coords,
            numbers=mol.structure.atomic_numbers,
            cell=mol.structure.lattice.matrix,  # 3 unit cell vectors
            pbc=True,
        )
        for mol in structures
    ]
    chunk_size = FLAGS.chunk
    args_list = [
        (
            molecules,
            seed,
            start,
            start + chunk_size,
            os.path.join(
                FLAGS.config.root_dir,
                f"fragments_{seed:02d}_{start:06d}_{start + chunk_size:06d}",
            ),
            None,
            # (FLAGS.config.nn_cutoff_min, FLAGS.config.nn_cutoff_max),
            FLAGS.config.min_n_nodes,
        )
        for seed in range(FLAGS.start_seed, FLAGS.end_seed)
        for start in range(0, len(structures), chunk_size)
    ]

    # Create a pool of processes, and apply generate_all_fragments to each tuple of arguments.
    tqdm.contrib.concurrent.process_map(_generate_all_fragments_wrapper, args_list, chunksize=128)

    # save structures and ase atoms of the structure that generated each fragment
    mp_ids_per_frag = []
    struct_per_frag = []
    ase_per_frag = []
    output_dir = FLAGS.config.root_dir
    for start in range(0, len(structures), chunk_size):
        end = start + chunk_size
        with open(os.path.join(output_dir, f"mol_indices_{start:06d}_{end:06d}.pkl"), "rb") as f:
            indices = pickle.load(f)
        for i in indices:
            mp_ids_per_frag.append(structures[start + i].material_id)
            struct_per_frag.append(structures[start + i].structure)
            ase_per_frag.append(ase.Atoms(
                positions = struct_per_frag[-1].cart_coords,
                numbers = struct_per_frag[-1].atomic_numbers,
                cell = struct_per_frag[-1].lattice.matrix,
                pbc = True
            ))
    with open(os.path.join(output_dir, "mp_ids_per_frag.pkl"), "wb") as f:
        pickle.dump(mp_ids_per_frag, f)
    with open(os.path.join(output_dir, "structures_per_frag.pkl"), "wb") as f:
        pickle.dump(struct_per_frag, f)
    with open(os.path.join(output_dir, "ase_atoms_per_frag.pkl"), "wb") as f:
        pickle.dump(ase_per_frag, f)

if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the training hyperparameter configuration.",
        lock_config=True,
    )
    flags.DEFINE_integer("start_seed", 0, "Start random seed.")
    flags.DEFINE_integer("end_seed", 1, "End random seed.")
    flags.DEFINE_integer("chunk", 50, "Number of molecules per fragment file.")
    flags.DEFINE_integer("start", None, "Start index.")
    flags.DEFINE_integer("end", None, "End index.")
    flags.DEFINE_float("nn_tolerance", 0.125, "NN tolerance (in Angstrom).")
    flags.DEFINE_float("max_radius", 3.0, "Max radius (in Angstrom).")

    app.run(main)
