import pickle
import sys
import tqdm

import jax
import jax.numpy as jnp
import jraph
import ml_collections
import yaml

import input_pipeline
import qm9
import models
import train


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


path = "/home/ameyad/spherical-harmonic-net/workdirs/v3/mace/interactions=1/l=3/channels=32"

with open(path + "/checkpoints/params.pkl", "rb") as f:
    params = pickle.load(f)
with open(path + "/config.yml", "rt") as config_file:
    config = yaml.unsafe_load(config_file)

assert config is not None
config = ml_collections.ConfigDict(config)

model = models.create_model(config, run_in_evaluation_mode=False)
apply_fn = jax.jit(model.apply)

epsilon = 0.125  # Angstroms
cutoff = 5.0  # Angstroms
filtering_threshold = 2.0  # Angstroms

molecules = qm9.load_qm9("qm9_data")

molecules_1k = molecules[-1000:]
seed = jax.random.PRNGKey(0)

losses = []

for i in tqdm.tqdm(range(1000)):
    molecule = molecules_1k[i]
    graph = input_pipeline.ase_atoms_to_jraph_graph(molecule, atomic_numbers, cutoff)
    frags = input_pipeline.generate_fragments(seed, graph, len(atomic_numbers), epsilon)
    frags = list(frags)

    frag = frags[-4]

    frag_unpadded = jraph.unpad_with_graphs(frag)
    molecules = jraph.unbatch(frag_unpadded)

    for j, mol in enumerate(molecules):
        preds = train.get_predictions(model, mol, seed)
        mol_loss = train.generation_loss(
            preds, mol, config.loss_kwargs.radius_rbf_variance
        )
        losses.append((mol_loss, i, j))

losses = sorted(losses)

with open("losses_1k.pkl", "wb") as f:
    f.write(losses)

with open("losses_1k_top_bot_25", "w") as f:
    f.write("Top 25:\n")
    for i in range(25):
        loss, frag_num, mol_num = losses[i]
        f.write(f"Fragment {frag_num} #{mol_num}:")
        f.write(f"total loss = {loss[0].tolist()[0]}\n")
        f.write(f"focus loss = {loss[1][0].tolist()[0]}\n")
        f.write(f"species loss = {loss[1][1].tolist()[0]}\n")
        f.write(f"position loss = {loss[1][2].tolist()[0]}\n")
        f.write("\n")

    f.write("\nBottom 25:\n")
    for i in range(25):
        loss, frag_num, mol_num = losses[-i]
        f.write(f"Fragment {frag_num} #{mol_num}:")
        f.write(f"total loss = {loss[0].tolist()[0]}\n")
        f.write(f"focus loss = {loss[1][0].tolist()[0]}\n")
        f.write(f"species loss = {loss[1][1].tolist()[0]}\n")
        f.write(f"position loss = {loss[1][2].tolist()[0]}\n")
        f.write("\n")
