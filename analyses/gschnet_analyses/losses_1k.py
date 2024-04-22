import jax
import jax.numpy as jnp
import jraph
import os
import pickle
import sys
import tensorflow as tf
import tqdm

sys.path.append("./analyses")
import analysis
import datatypes
import input_pipeline_tf
import train


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

tf.config.experimental.set_visible_devices([], "GPU")

interactions = int(sys.argv[1])
l = int(sys.argv[2])
channels = int(sys.argv[3])

atomic_numbers = jnp.array([1, 6, 7, 8, 9])
numbers_to_symbols = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
elements = list(numbers_to_symbols.values())

# covalent bond radii, in angstroms
element_radii = [0.32, 0.75, 0.71, 0.63, 0.64]


def get_numbers(species: jnp.ndarray):
    numbers = []
    for i in species:
        numbers.append(atomic_numbers[i])
    return jnp.array(numbers)


(
    config,
    best_state_train,
    best_state_eval,
    metrics_for_best_state,
) = analysis.load_from_workdir(
    f"/home/ameyad/spherical-harmonic-net/workdirs/v3/mace/interactions={interactions}/l={l}/channels={channels}",
    load_pickled_params=False,
)
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
    frag = jax.tree_util.tree_map(jnp.asarray, frag)

    frag_unpadded = jraph.unpad_with_graphs(frag)
    molecules = jraph.unbatch(frag_unpadded)

    for j, mol in enumerate(molecules):
        mol_padded = jraph.pad_with_graphs(mol, 32, 1024)
        preds, mol_loss = get_losses(mol_padded)
        mol_loss = jax.tree_util.tree_map(lambda x: x[0], mol_loss)
        losses.append((mol_loss, i, j, mol, jraph.unpad_with_graphs(preds)))

losses.sort(key=lambda x: abs(x[0][0]))

with open(
    f"loss_outputs/losses_1k_interactions={interactions}_l={l}_channels={channels}.pkl",
    "wb",
) as f:
    pickle.dump(losses, f)

first_nonstop = 0
for i, loss in enumerate(losses):
    if not loss[-2].globals.stop:
        first_nonstop = i
        break

with open(
    f"losses_1k_interactions={interactions}_l={l}_channels={channels}_topbot25.pkl",
    "wb",
) as f:
    pickle.dump(
        (losses[:25], losses[first_nonstop : first_nonstop + 25], losses[-25:]), f
    )
