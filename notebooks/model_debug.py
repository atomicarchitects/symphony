from ase.atoms import Atoms
from ase.io import write
from ase.visualize import view
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import tensorflow as tf

sys.path.append('..')
from analyses import analysis
from symphony import train, datatypes
from symphony.data import input_pipeline_tf
from symphony.models import utils


atomic_numbers = jnp.array([8, 14])
numbers_to_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'Si'}
elements = ['O', 'Si']

# covalent bond radii, in angstroms
element_radii = [0.63, 1.18]

def get_numbers(species: jnp.ndarray):
    numbers = []
    for i in species:
        numbers.append(atomic_numbers[i])
    return jnp.array(numbers)

tf.config.experimental.set_visible_devices([], "GPU")

workdir = '/data/NFS/potato/songk/spherical-harmonic-net/workdirs/silica_shuffled_matscipy_jan25/e3schnet_and_nequip/nn/max_targets_4'
best_eval, params, config = analysis.load_model_at_step(workdir, 'best', True, 359, 180)
config.root_dir = '/data/NFS/potato/songk/silica_shuffled_tetrahedra_nn3/nn/max_targets_4'
rng = jax.random.PRNGKey(config.rng_seed)
datasets = input_pipeline_tf.get_datasets(rng, config);

cutoff = 5.0
epsilon = 1e-4

frag_num = 0;

example_graph = next(datasets["test"].as_numpy_iterator())
frag = datatypes.Fragments.from_graphstuple(example_graph)
frag = jax.tree_map(jnp.asarray, frag)
frag_num += 1

frag_unpadded = jraph.unpad_with_graphs(frag)
molecules = jraph.unbatch(frag_unpadded);

mol_num = 4
mol = molecules[mol_num]
mol.globals.stop

species_list = mol.nodes.species.tolist()
positions_list = mol.nodes.positions.tolist()
target_species = mol.globals.target_species.tolist()[0]
target_positions = mol.globals.target_positions[0]
true_focus_index = 0

mol_padded = jraph.pad_with_graphs(
    mol,
    n_node=301,
    n_edge=3010,
    n_graph=2,
)

apply_fn = lambda padded_fragment, rng: best_eval.apply(
    params,
    rng,
    padded_fragment,
    1.0,
    1.0,
)

preds_eval = apply_fn(mol_padded, rng)
