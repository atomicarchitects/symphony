import ase
import ase.io
import jax
import jax.numpy as jnp
import jraph
import matscipy.neighbours
import numpy as np
import pickle
import random
import tqdm

from input_pipeline import ase_atoms_to_jraph_graph, generative_sequence
from qm9_nocache import load_qm9


def get_datasets(key: jax.random.PRNGKey, cutoff: float):
    """Creates training and test datasets from the larger QM9 dataset."""
    datasets = {}
    qm9_data = load_qm9("qm9_data")
    qm9_data_bar = tqdm.tqdm(qm9_data)
    subgraphs = []
    # collect graphs of partially-assembled molecules
    ct = 0
    for mol in qm9_data_bar:
        mol_graph = ase_atoms_to_jraph_graph(mol, cutoff)
        for subgraph in generative_sequence(key, mol_graph):
            subgraphs.append(subgraph)
        ct += 1
    random.shuffle(subgraphs, seed=0)
    return subgraphs


key = jax.random.PRNGKey(0)
qm9_data = get_datasets(key, 3.5)

pickle.dump(qm9_data, open("qm9_subgraphs_seed=0_cutoff=3.5.p", "w"))
