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

config, best_state_train, best_state_eval, metrics_for_best_state = analysis.load_from_workdir('workdirs/mace/interactions=2/l=3/channels=32', load_pickled_params=False)
rng = jax.random.PRNGKey(config.rng_seed)
rng, dataset_rng = jax.random.split(rng)
datasets = input_pipeline_tf.get_datasets(dataset_rng, config)

cutoff = 5.0
epsilon = 1e-4

losses = []

for i in tqdm.tqdm(range(1000)):
    graph = next(datasets["test"].as_numpy_iterator())
    frag = datatypes.Fragment.from_graphstuple(graph)
    frag = jax.tree_map(jnp.asarray, frag)

    frag_unpadded = jraph.unpad_with_graphs(frag)
    molecules = jraph.unbatch(frag_unpadded)

    for j, mol in enumerate(molecules):
        preds = train.get_predictions(best_state_train, mol, rng)
        mol_loss = train.generation_loss(preds, mol, config.loss_kwargs.radius_rbf_variance)
        losses.append((mol_loss, i, j))

losses = sorted(losses)

with open('losses_1k.p', 'wb') as f:
    f.write(losses)

with open('losses_1k_top_bot_25', 'w') as f:
    f.write('Top 25:\n')
    for i in range(25):
        loss, frag_num, mol_num = losses[i]
        f.write(f'Fragment {frag_num} #{mol_num}:')
        f.write(f'total loss = {loss[0].tolist()[0]}\n')
        f.write(f'focus loss = {loss[1][0].tolist()[0]}\n')
        f.write(f'species loss = {loss[1][1].tolist()[0]}\n')
        f.write(f'position loss = {loss[1][2].tolist()[0]}\n')
        f.write('\n')

    f.write('\nBottom 25:\n')
    for i in range(25):
        loss, frag_num, mol_num = losses[-i]
        f.write(f'Fragment {frag_num} #{mol_num}:')
        f.write(f'total loss = {loss[0].tolist()[0]}\n')
        f.write(f'focus loss = {loss[1][0].tolist()[0]}\n')
        f.write(f'species loss = {loss[1][1].tolist()[0]}\n')
        f.write(f'position loss = {loss[1][2].tolist()[0]}\n')
        f.write('\n')
