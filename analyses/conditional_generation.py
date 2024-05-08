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
from symphony.data.datasets import perov
from configs.root_dirs import get_root_dir
from symphony import datatypes


def get_fragment_list(mols: Sequence[datatypes.Structures], num_mols: int):
    fragments = []
    for i in range(num_mols):
        mol = mols[i]
        # offset by 1
        filter_ON = (mol.nodes.species == 6) | (mol.nodes.species == 7)
        fragment = ase.Atoms(
            # all perovskites in perov5 contain either O or N as negative ion
            positions=mol.nodes.positions[~filter_ON, :],
            numbers=mol.nodes.species[~filter_ON],
            cell=mol.globals.cell.reshape(3, 3),
            pbc=True
        )
        fragments.append(fragment)
    return fragments


def main(unused_argv: Sequence[str]):
    radial_cutoff = 5.0
    beta_species = 1.0
    beta_position = 1.0
    step = flags.FLAGS.step
    num_seeds_per_chunk = 1
    max_num_atoms = 35
    num_mols = 20
    avg_neighbors_per_atom = 10

    atomic_numbers = np.array([
            3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
            32, 33, 37, 38, 39, 40, 41, 42, 44, 45, 46, 
            47, 48, 49, 50, 51, 52, 55, 56, 57, 72, 73, 
            74, 75, 76, 77, 78, 79, 80, 81, 82, 83
        ])

    all_mols = perov.load_perov(get_root_dir('perov5'), supercell=False)
    mols_by_split = {"train": all_mols['train'][:num_mols], "test": all_mols['test'][-num_mols:]}

    for split, split_mols in mols_by_split.items():
        # Ensure that the number of molecules is a multiple of num_seeds_per_chunk.
        mol_list = get_fragment_list(split_mols, num_mols)
        mol_list = mol_list[
            : num_seeds_per_chunk * (len(split_mols) // num_seeds_per_chunk)
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
    app.run(main)
