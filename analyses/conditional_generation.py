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
import analyses.generate_molecules_intermediates as generate_molecules_intermediates
from symphony.data import linker


def main(unused_argv: Sequence[str]):
    beta_species = 1.0
    beta_position = 1.0
    step = flags.FLAGS.step
    num_seeds_per_chunk = 1
    max_num_atoms = 60
    max_num_steps = 35
    # num_mols = 1
    num_mols = 50

    all_mols = linker.load_linker("/home/songk/symphony-linker/linker_data")
    mols_by_split = {"train": all_mols['frags'][:num_mols], "test": all_mols['frags'][-num_mols:]}

    for split, split_mols in mols_by_split.items():
        # Ensure that the number of molecules is a multiple of num_seeds_per_chunk.
        mol_list = split_mols[
            : num_seeds_per_chunk * (len(split_mols) // num_seeds_per_chunk)
        ]
        print(f"Number of fragments for {split}: {len(mol_list)}")

        args = [
            flags.FLAGS.workdir,
            os.path.join(flags.FLAGS.outputdir, split),
            beta_species,
            beta_position,
            step,
            len(mol_list),
            num_seeds_per_chunk,
            mol_list,
            max_num_atoms,
            max_num_steps,
        ]

        if flags.FLAGS.store_intermediates:
            args = args[:6] + args[7:]
            gen_mol_list = generate_molecules_intermediates.generate_molecules(*args)
        else:
            args += [flags.FLAGS.visualize]
            gen_mol_list = generate_molecules.generate_molecules(*args)


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
    flags.DEFINE_bool(
        "store_intermediates",
        False,
        "Whether to store intermediates.",
    )
    app.run(main)
