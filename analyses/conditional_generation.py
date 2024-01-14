# Imports
from typing import *
import ase
import ase.db
import ase.io
import os
import numpy as np
import re
import sys
import tensorflow as tf

from absl import flags, app
import analyses.generate_molecules as generate_molecules
from symphony.data import qm9


def get_fragment_list(mols: Sequence[ase.Atoms], num_mols: int):
    fragments = []
    for i in range(num_mols):
        mol = mols[i]
        num_atoms = len(mol)
        for j in range(num_atoms):
            if mol.numbers[j] == 1:
                fragment = ase.Atoms(
                    positions=np.vstack([mol.positions[:j], mol.positions[j + 1 :]]),
                    numbers=np.concatenate([mol.numbers[:j], mol.numbers[j + 1 :]]),
                )
                fragments.append(fragment)
    return fragments


def main(unused_argv: Sequence[str]):
    beta_species = 1.0
    beta_position = 1.0
    step = "best"
    num_seeds_per_chunk = 25
    max_num_atoms = 35
    num_mols = 1000

    all_mols = qm9.load_qm9("../qm9_data", use_edm_splits=True, check_molecule_sanity=False)
    mols_by_split = {"train": all_mols[:num_mols], "test": all_mols[-num_mols:]}

    for split, split_mols in mols_by_split.items():
        # Ensure that the number of molecules is a multiple of num_seeds_per_chunk.
        mol_list = get_fragment_list(split_mols, num_mols)
        mol_list = split_mols[
            : num_seeds_per_chunk * (len(split_mols) // num_seeds_per_chunk)
        ]
        print(f"Number of fragments for {split}: {len(mol_list)}")

        gen_mol_list = generate_molecules.generate_molecules(
            flags.FLAGS.workdir,
            os.path.join(flags.FLAGS.outputdir, split),
            beta_species,
            beta_position,
            step,
            len(mol_list),
            num_seeds_per_chunk,
            mol_list,
            max_num_atoms,
            flags.FLAGS.visualize,
        )


if __name__ == "__main__":
    flags.DEFINE_string(
        "workdir",
        "/data/NFS/potato/songk/spherical-harmonic-net/workdirs/",
        "Workdir for model.",
    )
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "conditional_generation", "analysed_workdirs"),
        "Directory where molecules should be saved.",
    )
    flags.DEFINE_bool(
        "visualize",
        False,
        "Whether to visualize the generation process step-by-step.",
    )
    app.run(main)
