# Imports
from typing import *
import ase
import ase.db
import ase.io
import os
import numpy as np
import sys

import analyses.generate_molecules as generate_molecules
from symphony.data import qm9



workdir = "/home/ameyad/spherical-harmonic-net/workdirs/qm9_bessel_embedding_attempt6_edm_splits/e3schnet_and_nequip/interactions=3/l=5/position_channels=2/channels=64"
outputdir = "conditional_generation"
beta_species = 1.0
beta_position = 1.0
step = "4950000"
num_seeds_per_chunk = 25
max_num_atoms = 35
visualize = False
num_mols = 1000

all_mols = qm9.load_qm9("../qm9_data", use_edm_splits=True, check_molecule_sanity=False)
test_mols = all_mols[-num_mols:]
train_mols = all_mols[:num_mols]


def get_fragment_list(mols: Sequence[ase.Atoms], num_mols: int):
    fragments = []
    for i in range(num_mols):
        mol = mols[i]
        num_atoms = len(mol)
        for j in range(num_atoms):
            fragment = ase.Atoms(
                positions=np.vstack([mol.positions[:j], mol.positions[j + 1 :]]),
                numbers=np.concatenate([mol.numbers[:j], mol.numbers[j + 1 :]]),
            )
            fragments.append(fragment)
    return fragments

# Ensure that the number of molecules is a multiple of num_seeds_per_chunk.
mol_list = get_fragment_list(train_mols, num_mols)
mol_list = mol_list[:num_seeds_per_chunk * (len(mol_list) // num_seeds_per_chunk)]
print(f"Number of molecules: {len(mol_list)}")

gen_mol_list = generate_molecules.generate_molecules(
    workdir,
    os.path.join(outputdir, "train"),
    beta_species,
    beta_position,
    step,
    len(mol_list),
    num_seeds_per_chunk,
    mol_list,
    max_num_atoms,
    visualize,
)

# Ensure that the number of molecules is a multiple of num_seeds_per_chunk.
mol_list = get_fragment_list(test_mols, num_mols)
mol_list = mol_list[:num_seeds_per_chunk * (len(mol_list) // num_seeds_per_chunk)]
print(f"Number of molecules: {len(mol_list)}")

gen_mol_list = generate_molecules.generate_molecules(
    workdir,
    os.path.join(outputdir, "test"),
    beta_species,
    beta_position,
    step,
    len(mol_list),
    num_seeds_per_chunk,
    mol_list,
    max_num_atoms,
    visualize,
)
