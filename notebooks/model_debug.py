import ase
import ase.io
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import tqdm

sys.path.append('..')
from analyses import analysis
from symphony import train, datatypes
from symphony.data import input_pipeline_tf

tf.config.experimental.set_visible_devices([], "GPU")

workdir = '/data/NFS/potato/songk/spherical-harmonic-net/workdirs/silica-nequip-heavy-first-jan4'
config, best_state_train, best_state_eval, metrics_for_best_state = analysis.load_from_workdir(workdir)
rng = jax.random.PRNGKey(config.rng_seed)
datasets = input_pipeline_tf.get_datasets(rng, config)

cutoff = 5.0
epsilon = 1e-4

max_neighbors = 0
for split in ['train', 'val', 'test']:
    print(f"Checking {split}:")
    for graphs in tqdm.tqdm(datasets[split].as_numpy_iterator()):
        frags = jax.tree_map(jnp.asarray, graphs)
        frag_unpadded = jraph.unpad_with_graphs(frags)
        molecules = jraph.unbatch(frag_unpadded)
        for mol in molecules:
            if jnp.linalg.norm(mol.edges.relative_positions, axis=-1).min() < 1e-5:
                print(jnp.linalg.norm(mol.edges.relative_positions, axis=-1))
            if mol.globals.stop[0]:
                continue
            max_neighbors = max(max_neighbors, mol.receivers[mol.senders == 0].shape[0])

print("Max # of neighbors for focus:", max_neighbors)