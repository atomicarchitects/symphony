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
from symphony.data.datasets import tmqm
from configs.root_dirs import get_root_dir
from symphony import datatypes


workdir = "/home/ameyad/spherical-harmonic-net/workdirs/qm9_bessel_embedding_attempt6_edm_splits/e3schnet_and_nequip/interactions=3/l=5/position_channels=2/channels=64"
outputdir = "conditional_generation"


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
    radial_cutoff = 5.0
    beta_species = 1.0
    beta_position = 1.0
    step = flags.FLAGS.step
    num_seeds_per_chunk = 1
    max_num_atoms = 50
    num_mols = 500
    avg_neighbors_per_atom = 32

    atomic_numbers = np.arange(1, 81)

    all_mols = tmqm.load_tmqm("../tmqm_data")
    mols_by_split = {"train": all_mols[:num_mols], "test": all_mols[-num_mols:]}

    for split, split_mols in mols_by_split.items():
        # Ensure that the number of molecules is a multiple of num_seeds_per_chunk.
        mol_list = get_fragment_list(split_mols, num_mols)
        mol_list = mol_list[
            : num_seeds_per_chunk * (len(mol_list) // num_seeds_per_chunk)
        ]
        print(f"Number of fragments for {split}: {len(mol_list)}")

        gen_mol_list = generate_molecules.generate_molecules_from_workdir(
            flags.FLAGS.workdir,
            os.path.join(flags.FLAGS.outputdir, split),
            radial_cutoff,
            beta_species,
            beta_position,
            step,
            flags.FLAGS.steps_for_weight_averaging,
            len(mol_list),
            num_seeds_per_chunk,
            mol_list,
            max_num_atoms,
            avg_neighbors_per_atom,
            atomic_numbers,
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
    flags.DEFINE_string(
        "step",
        "best",
        "Step number to load model from. The default corresponds to the best model.",
    )
    flags.DEFINE_list(
        "steps_for_weight_averaging",
        None,
        "Steps to average parameters over. If None, the model at the given step is used.",
    )
    flags.DEFINE_bool(
        "seed_structure",
        False,
        "Add initial atom of the missing element to structure"
    )
    app.run(main)
