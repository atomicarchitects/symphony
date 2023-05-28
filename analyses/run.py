import os
import subprocess
from ase.db import connect
import numpy as np
import pandas as pd
import pickle
import tqdm
import jax
import jax.numpy as jnp
import jraph
import ase
import sys
sys.path.append('..')

import analysis
import check_valence
import input_pipeline
import datatypes
import models

def predict_atom_removals(
    workdir: str,
    beta: float,
    step: int,
    molecule: ase.Atoms,
    seed: int,
):
    """Generates visualizations of the predictions when removing each atom from a molecule."""
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )

    # Remove the target atoms from the molecule.
    molecules_with_target_removed = []
    fragments = []
    for target in range(len(molecule)):
        molecule_with_target_removed = ase.Atoms(
            positions=np.concatenate(
                [molecule.positions[:target], molecule.positions[target + 1 :]]
            ),
            numbers=np.concatenate(
                [molecule.numbers[:target], molecule.numbers[target + 1 :]]
            ),
        )
        fragment = input_pipeline.ase_atoms_to_jraph_graph(
            molecule_with_target_removed,
            analysis.ATOMIC_NUMBERS,
            config.nn_cutoff,
        )

        molecules_with_target_removed.append(molecule_with_target_removed)
        fragments.append(fragment)

    # We don't actually need a PRNG key, since we're not sampling.
    print("Computing predictions...")

    rng = jax.random.PRNGKey(seed)
    preds = jax.jit(model.apply)(params, rng, jraph.batch(fragments), beta)
    preds = jax.tree_map(np.asarray, preds)
    preds = jraph.unbatch(preds)
    print("Predictions computed.")

    # Loop over all possible targets.
    print("Visualizing predictions...")
    figs = []
    preds_list = []
    for target in range(len(molecule)):
        # We have to remove the batch dimension.
        # Also, correct the focus indices due to batching.
        pred = preds[target]._replace(
            globals=jax.tree_map(lambda x: np.squeeze(x, axis=0), preds[target].globals)
        )
        corrected_focus_indices = pred.globals.focus_indices - sum(
            p.n_node.item() for i, p in enumerate(preds) if i < target
        )
        pred = pred._replace(
            globals=pred.globals._replace(focus_indices=corrected_focus_indices)
        )
        preds_list.append(pred)

    return molecules_with_target_removed, preds_list

def append_predictions(
        molecule: ase.Atoms, pred: datatypes.Predictions
    ) -> ase.Atoms:
    focus = pred.globals.focus_indices
    pos_focus = molecule.positions[focus]
    pos_rel = pred.globals.position_vectors

    new_species = jnp.array(
        models.ATOMIC_NUMBERS[pred.globals.target_species.item()]
    )
    new_position = pos_focus + pos_rel

    return ase.Atoms(
        positions=jnp.concatenate(
            [molecule.positions, new_position[None, :]], axis=0
        ),
        numbers=jnp.concatenate([molecule.numbers, new_species[None]], axis=0),
    )

workdir = '/Users/songk/atomicarchitects/spherical_harmonic_net/workdirs'

interactions = 4
l = 5
params = {
    6: {
        'nequip': {'channels': [128], 'steps': ['best']},
        # 'nequip': {'channels': [32, 64, 128], 'steps': ['best', 'best', 'best']},
        # 'nequip-l2': {'channels': [32, 64, 128], 'steps': [885000, 675000, 375000]},
    },
    7: {
        'nequip-l2': {'channels': [32], 'steps': ['best']},
    }
}

molecules = []
np.random.seed(0)
indices = np.random.choice(np.arange(53568, 133885), 64, replace=False)
with connect('../qm9_data/qm9-all.db') as conn:
    for i, row in enumerate(conn.select()):
        if i in indices: molecules.append(row.toatoms())

stats = {}

for version in [6]:
    for model in params[version]:
        for channels, step in tqdm.tqdm(zip(params[version][model]['channels'], params[version][model]['steps'])):
            for beta in [1, 10, 100, 1000]:
                for init in ['CH3', 'C6H5']:
                    model_path = os.path.join(
                        workdir,
                        f'v{version}',
                        model,
                        f'interactions={interactions}',
                        f'l={l}',
                        f'channels={channels}'
                        )
                    mol_path = os.path.join(
                        'tmp/molecules/generated/',
                        f'v{version}',
                        model,
                        f'interactions={interactions}',
                        f'l={l}',
                        f'channels={channels}',
                        f'beta={beta:.1f}',
                        f'step={step}',
                    )

                    subprocess.run([
                        'python',
                        'generate_molecules.py',
                        f'--workdir={model_path}',
                        f'--outputdir=tmp',
                        '--visualize=True',
                        # f'--step={step}',
                        f'--beta={beta}',
                        f'--init=molecules/downloaded/{init}.xyz'
                    ])

                    subprocess.run([
                        'python',
                        'qm9_filter_generated.py',
                        mol_path,
                        f'--model_path={model_path}',
                        '--data_path=../qm9_data/qm9-all.db',
                        '--threads=0',
                        f'--init={init}'
                    ])

#                     num_total_mols = 0
#                     num_valid_mols = 0
#                     num_correct_mols = 0
#                     num_valid_atoms = 0
#                     num_total_atoms = 0
#                     for molecule in molecules:
#                         molecules_with_target_removed, preds = predict_atom_removals(
#                             workdir=model_path,
#                             beta=1.0,
#                             step=step,
#                             molecule=molecule,
#                             seed=0)
#                         for mol, pred in zip(molecules_with_target_removed, preds):
#                             reconstructed = append_predictions(mol, pred)
                            
#                             valid_mol, valid_atoms = check_valence.check_valence(reconstructed)

#                             # update stat counts
#                             num_total_mols += 1
#                             num_total_atoms += len(mol.numbers)
#                             num_valid_atoms += valid_atoms
#                             if sorted(molecule.numbers) == sorted(reconstructed.numbers):
#                                 num_valid_mols += int(valid_mol)
#                                 num_correct_mols += 1
#                     stats[(version, model, channels, step)] = (num_valid_mols, num_correct_mols, num_total_mols, num_valid_atoms, num_total_atoms)
# pickle.dump(stats, open('stats.pkl', 'wb'))
