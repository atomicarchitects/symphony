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
import random

from symphony.data import fragments
from symphony.data import input_pipeline, matproj

from configs import root_dirs

FLAGS = flags.FLAGS


def generate_all_fragments(
    molecules: List[ase.Atoms],
    seed: int,
    start: int,
    end: int,
    output_dir: str,
    mode: str,
    cutoff: float,
    min_n_nodes: float,
    max_targets_per_graph: int,
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
        "target_positions": tf.TensorSpec(
            shape=(1, max_targets_per_graph, 3), dtype=tf.float32
        ),
        "target_position_mask": tf.TensorSpec(
            shape=(1, max_targets_per_graph), dtype=tf.float32
        ),
        "target_species": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "cell": tf.TensorSpec(shape=(1, 3, 3), dtype=tf.float32),
        # n_node and n_edge
        "n_node": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        "n_edge": tf.TensorSpec(shape=(1,), dtype=tf.int32),
    }

    mol_indices = []  # mol ndx matching up with each fragment

    def generator():
        for mol_ndx, mol in tqdm.tqdm(enumerate(molecules)):
            graph = input_pipeline.ase_atoms_to_jraph_graph(
                mol, atomic_numbers, cutoff=cutoff, periodic=True
            )
            assert np.equal(graph.senders, graph.receivers).sum() == 0, "self edges!"
            if FLAGS.tetrahedra_only:
                frags = fragments.generate_silica_fragments(
                rng,
                graph,
                atomic_numbers,
                FLAGS.nn_tolerance,
                FLAGS.max_radius,
                mode,
                heavy_first=FLAGS.config.heavy_first,
                max_targets_per_graph=max_targets_per_graph,
            )
            else:
                frags = fragments.generate_fragments(
                    rng,
                    graph,
                    atomic_numbers.shape[0],
                    FLAGS.nn_tolerance,
                    FLAGS.max_radius,
                    mode,
                    heavy_first=FLAGS.config.heavy_first,
                    max_targets_per_graph=max_targets_per_graph,
                    periodic=True,
                    num_fragments=FLAGS.num_frags_per_graph,
                )

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
                    "target_position_mask": frag.globals.target_position_mask.astype(
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


def _generate_all_fragments_wrapper(args):
    """Dummy wrapper to allow parallelization."""
    return generate_all_fragments(*args)


def main(unused_argv) -> None:
    del unused_argv

    logging.set_verbosity(logging.INFO)
    logging.set_stderrthreshold(logging.INFO)

    # Create a list of arguments to pass to generate_all_fragments
    if FLAGS.structure_file is not None:
        with open(FLAGS.structure_file, "rb") as f:
            molecules = pickle.load(f)
    else:
        structures = matproj.get_materials({"elements": ["O", "Si"], "num_elements": (2, 2)})
        molecules = [
            ase.Atoms(
                positions=mol.structure.cart_coords,
                numbers=mol.structure.atomic_numbers,
                cell=mol.structure.lattice.matrix,  # 3 unit cell vectors
                pbc=True,
            )
            for mol in structures
        ]
    if FLAGS.shuffle:
        random.seed(FLAGS.shuffle_seed)
        random.shuffle(molecules)
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.mode, f"max_targets_{FLAGS.max_targets_per_graph}")
    chunk_size = FLAGS.chunk
    args_list = [
        (
            molecules,
            seed,
            start,
            start + chunk_size,
            os.path.join(
                output_dir,
                f"fragments_{seed:02d}_{start:06d}_{start + chunk_size:06d}",
            ),
            FLAGS.mode,
            FLAGS.nn_cutoff,
            FLAGS.min_n_nodes,
            FLAGS.max_targets_per_graph,
        )
        for seed in range(FLAGS.start_seed, FLAGS.end_seed)
        for start in range(0, len(molecules), chunk_size)
    ]

    # Create a pool of processes, and apply generate_all_fragments to each tuple of arguments.
    tqdm.contrib.concurrent.process_map(_generate_all_fragments_wrapper, args_list, chunksize=128)


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
    flags.DEFINE_string("mode", "radius", "Fragmentation mode.")
    flags.DEFINE_string(
        "output_dir", "/data/NFS/potato/songk/silica_fragments/", "Output directory."
    )
    flags.DEFINE_bool(
        "tetrahedra_only", False, "Whether to remove single SiO4 tetrahedra only."
    )
    flags.DEFINE_float("nn_tolerance", 0.125, "NN tolerance (in Angstrom).")
    flags.DEFINE_float("max_radius", 2.03, "Max radius (in Angstrom).")
    flags.DEFINE_integer(
        "max_targets_per_graph", 1, "Max num of targets per focus atom."
    )
    flags.DEFINE_string(
        "structure_file", None, "Location of cached structures."
    )
    flags.DEFINE_integer(
        "num_frags_per_graph", 0, "Number of fragments per structure. If 0, generate all fragments."
    )
    flags.DEFINE_integer(
        "min_n_nodes", 30, "Minimum number of nodes per graph."
    )
    flags.DEFINE_bool(
        "shuffle", False, "Whether to shuffle the dataset."
    )
    flags.DEFINE_integer(
        "shuffle_seed", 0, "Seed to use when shuffling dataset."
    )
    flags.DEFINE_float("nn_cutoff", 5.0, "NN cutoff (in Angstrom).")

    app.run(main)
