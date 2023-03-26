import jax
import jax.numpy as jnp
import jraph
import pickle
import sys
import tqdm

sys.path.append('./analyses')
import analysis
import datatypes
import input_pipeline_tf
import train


interactions=int(sys.argv[1])
l=int(sys.argv[2])
channels=int(sys.argv[3])


atomic_numbers = jnp.array([1, 6, 7, 8, 9])
numbers_to_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
elements = list(numbers_to_symbols.values())

# covalent bond radii, in angstroms
element_radii = [0.32, 0.75, 0.71, 0.63, 0.64]

def get_numbers(species: jnp.ndarray):
    numbers = []
    for i in species:
        numbers.append(atomic_numbers[i])
    return jnp.array(numbers)

config, best_state_train, best_state_eval, metrics_for_best_state = analysis.load_from_workdir(f"/home/ameyad/spherical-harmonic-net/workdirs/v3/mace/interactions={interactions}/l={l}/channels={channels}", load_pickled_params=False)
rng = jax.random.PRNGKey(config.rng_seed)
rng, dataset_rng = jax.random.split(rng)
datasets = input_pipeline_tf.get_datasets(dataset_rng, config)

cutoff = 5.0
epsilon = 1e-4


losses = []

@jax.jit
def get_losses(mol):
    preds = train.get_predictions(best_state_train, mol, rng)
    mol_loss = train.generation_loss(preds, mol, config.loss_kwargs.radius_rbf_variance)
    return preds, mol_loss

for i in tqdm.tqdm(range(1000)):
    graph = next(datasets["test"].as_numpy_iterator())
    frag = datatypes.Fragments.from_graphstuple(graph)
    frag = jax.tree_map(jnp.asarray, frag)

    frag_unpadded = jraph.unpad_with_graphs(frag)
    molecules = jraph.unbatch(frag_unpadded)

    for j, mol in enumerate(molecules):
        mol_padded = jraph.pad_with_graphs(mol, 32, 1024)
        preds, mol_loss = get_losses(mol_padded)
        mol_loss = jax.tree_util.tree_map(lambda x: x[0], mol_loss)
        losses.append((mol_loss, i, j, mol, jraph.unpad_with_graphs(preds)))

losses = sorted(losses)

with open(f'loss_outputs/losses_1k_interactions={interactions}_l={l}_channels={channels}.pkl', 'wb') as f:
    pickle.dump(losses, f)

