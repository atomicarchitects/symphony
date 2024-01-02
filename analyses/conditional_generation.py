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
sys.path.append('../configs/silica/allegro.py')

from absl import flags, app
import analyses.generate_molecules as generate_molecules
from symphony.data import input_pipeline_tf

import configs.silica.allegro as allegro

def main(unused_argv: Sequence[str]):
    beta_species = 1.0
    beta_position = 1.0
    step = "best"
    num_seeds_per_chunk = 1
    max_num_atoms = 200  # ?
    visualize = False
    num_mols = 20
    config = allegro.get_config()

    mols_by_split = {"train": [], "test": []}

    # Root directory of the dataset.
    filenames = sorted(os.listdir(config.root_dir))
    filenames = [
        os.path.join(config.root_dir, f)
        for f in filenames
        if f.startswith("fragments_")
    ]
    if len(filenames) == 0:
        raise ValueError(f"No files found in {config.root_dir}.")

    # Partition the filenames into train, val, and test.
    def filter_by_molecule_number(
        filenames: Sequence[str], start: int, end: int
    ) -> List[str]:
        def filter_file(filename: str, start: int, end: int) -> bool:
            filename = os.path.basename(filename)
            _, file_start, file_end = [int(val) for val in re.findall(r"\d+", filename)]
            return start <= file_start and file_end <= end

        return [f for f in filenames if filter_file(f, start, end)]

    # Number of molecules for training can be smaller than the chunk size.
    chunk_size = int(filenames[0].split("_")[-1])
    train_on_split_smaller_than_chunk = config.get("train_on_split_smaller_than_chunk")
    if train_on_split_smaller_than_chunk:
        train_molecules = (0, chunk_size)
    else:
        train_molecules = config.train_molecules
    files_by_split = {
        "train": filter_by_molecule_number(filenames, *train_molecules),
        "test": filter_by_molecule_number(filenames, *config.test_molecules),
    }

    element_spec = tf.data.Dataset.load(filenames[0]).element_spec
    for split, files in files_by_split.items():
        dataset_split = tf.data.Dataset.from_tensor_slices(files)
        dataset_split = dataset_split.interleave(
            lambda x: tf.data.Dataset.load(x, element_spec=element_spec),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        dataset_split = dataset_split.map(
            input_pipeline_tf._convert_to_graphstuple,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        mols_by_split[split] = list(dataset_split.as_numpy_iterator())

    mols_by_split['train'] = mols_by_split['train'][:num_mols]
    mols_by_split['test'] = mols_by_split['test'][-num_mols:]
    print([g.n_node for g in mols_by_split['train']])
    print([g.n_node for g in mols_by_split['test']])

    for split, split_mols in mols_by_split.items():
        # Ensure that the number of molecules is a multiple of num_seeds_per_chunk.
        mol_list = split_mols[:num_seeds_per_chunk * (len(split_mols) // num_seeds_per_chunk)]
        print(f"Number of molecules: {len(mol_list)}")

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
            visualize,
            filetype="cif"
        )

if __name__ == "__main__":
    flags.DEFINE_string(
        "workdir",
        "/data/NFS/potato/songk/spherical-harmonic-net/workdirs/silica-allegro-200k-train-steps-all-dec29",
        "Workdir for model."
    )
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "conditional_generation", "analysed_workdirs"),
        "Directory where molecules should be saved.",
    )
    app.run(main)