"""Generates molecules from a trained model."""

from typing import List, Sequence

import os
import pickle
import sys

from absl import flags
from absl import app
import ase
import ase.data
import ase.io
import ase.visualize
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import tqdm
import yaml
import chex

sys.path.append('..')

import datatypes
import input_pipeline
import train
import models

FLAGS = flags.FLAGS


def generate_molecules(workdir: str, outputdir: str, beta: float):
    """Generates molecules from a trained model at the given workdir."""

    with open(workdir + "/checkpoints/params.pkl", 'rb') as f:
        params = pickle.load(f)
    with open(workdir + "/config.yml", "rt") as config_file:
        config = yaml.unsafe_load(config_file)

    assert config is not None
    config = ml_collections.ConfigDict(config)
    
    index = workdir.find("workdirs") + len("workdirs/")
    name = workdir[index:]

    model = train.create_model(config, run_in_evaluation_mode=True)
    apply_fn = jax.jit(model.apply)

    def get_predictions(frag: jraph.GraphsTuple, rng: chex.PRNGKey) -> datatypes.Predictions:
        frags = jraph.pad_with_graphs(frag, 32, 1024, 2)
        preds = apply_fn(params, rng, frags, beta)
        pred = jraph.unpad_with_graphs(preds)
        return pred

    def append_predictions(molecule: ase.Atoms, pred: datatypes.Predictions) -> ase.Atoms:
        focus = pred.globals.focus_indices.squeeze(0)
        pos_focus = molecule.positions[focus]
        pos_rel = pred.globals.position_vectors.squeeze(0)

        new_species = jnp.array(
            models.ATOMIC_NUMBERS[pred.globals.target_species.squeeze(0).item()]
        )
        new_position = pos_focus + pos_rel

        return ase.Atoms(
            positions=jnp.concatenate([molecule.positions, new_position[None, :]], axis=0),
            numbers=jnp.concatenate([molecule.numbers, new_species[None]], axis=0),
        )

    molecules = []

    # Generate with different seeds.
    for seed in tqdm.tqdm(range(64)):
        molecule = ase.Atoms(
            positions=jnp.array([[0., 0., 0.]]),
            numbers=jnp.array([6]),
        )

        rng = jax.random.PRNGKey(seed)
        for _ in range(31):
            step_rng, rng = jax.random.split(rng)
            frag = input_pipeline.ase_atoms_to_jraph_graph(
                molecule, models.ATOMIC_NUMBERS, config.nn_cutoff
            )
            pred = get_predictions(frag, step_rng)

            stop = pred.globals.stop.squeeze(0).item()
            if stop:
                break

            molecule = append_predictions(molecule, pred)

        if molecule.numbers.shape[0] < 32:
            molecules.append(molecule)

    # Save molecules.
    outputdir = os.path.join(outputdir, name, f"beta={beta}", "generated")
    os.makedirs(outputdir, exist_ok=True)
    for seed, molecule in enumerate(molecules):
        ase.io.write(f"{outputdir}/molecule_{seed}.xyz", molecule)
    ase_to_mol_dict(molecules, file_name=f"{outputdir}/generated_molecules.mol_dict")


def ase_to_mol_dict(molecules: List[ase.Atoms], save=True, file_name=None):
    '''from G-SchNet: https://github.com/atomistic-machine-learning/G-SchNet'''

    generated = (
        {}
    )
    for mol in molecules:
        l = mol.get_atomic_numbers().shape[0]
        if l not in generated:
            generated[l] = {
                "_positions": np.array([mol.get_positions()]),
                "_atomic_numbers": np.array([mol.get_atomic_numbers()]),
            }
        else:
            generated[l]["_positions"] = np.append(
                generated[l]["_positions"],
                np.array([mol.get_positions()]),
                0,
            )
            generated[l]["_atomic_numbers"] = np.append(
                generated[l]["_atomic_numbers"],
                np.array([mol.get_atomic_numbers()]),
                0,
            )

    if save:
        with open(file_name, "wb") as f:
            pickle.dump(generated, f)

    return generated


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = FLAGS.workdir
    outputdir = FLAGS.outputdir
    beta = FLAGS.beta

    generate_molecules(workdir, outputdir, beta)


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for molecules.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses", "molecules"),
        "Directory where molecules should be saved.",
    )
    flags.DEFINE_float("beta", 1.0, "Inverse temperature value for sampling.")

    flags.mark_flags_as_required(["workdir"])
    app.run(main)