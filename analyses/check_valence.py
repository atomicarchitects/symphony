##########################################################
# From G-SchNet repository                               #
# https://github.com/atomistic-machine-learning/G-SchNet #
##########################################################

import argparse
import os
import pickle
import numpy as np
import pandas as pd

from analyses.analysis import construct_pybel_mol


def get_parser():
    """Setup parser for command line arguments"""
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "mol_path",
        help="Path to generated molecules in .mol_dict format, "
        'a database called "generated_molecules.db" with the '
        "filtered molecules along with computed statistics "
        '("generated_molecules_statistics.pkl") will be '
        "stored in the same directory as the input file/s "
        "(if the path points to a directory, all .mol_dict "
        "files in the directory will be merged and filtered "
        "in one pass)",
    )

    return main_parser


def check_valence(
    mol,
    valence=[0, 1, 0, 0, 0, 0, 4, 3, 2, 1],
):
    """
    Assess whether a molecule meets valence constraints or not
    (i.e. all of their atoms have the correct number of bonds).

    Args:
        mol (ase.Atoms): an ASE molecule
        valence (numpy.ndarray): list of valenc of each atom type where the index in
            the list corresponds to the type (e.g. [0, 1, 0, 0, 0, 0, 4, 3, 2, 1] for
            qm9 molecules as H=type 1 has valency of 1, C=type 6 has valency of 4,
            N=type 7 has valency of 3 etc.)
        print_file (bool, optional): set True to suppress printing of progress string
            (default: True)
        prog_str (str, optional): specify a custom progress string (default: None)

    Returns:
        valid_mol (bool): True if molecule has passed the valence check, False otherwise
        valid_atoms (int): the number of atoms in each molecule that pass the valence check
    """
    n_atoms = len(mol.numbers)
    valid_atoms = 0
    pybel_mol = construct_pybel_mol(mol)

    for atom in pybel_mol.atoms:
        if atom.OBAtom.GetExplicitValence() == valence[atom.atomicnum]:
            valid_atoms += 1
    
    return valid_atoms == n_atoms, valid_atoms


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    #     d = res[n_atoms]
    #     all_pos = d[Properties.R]
    #     all_numbers = d[Properties.Z]
    #     n_mols = len(all_pos)

    #     # check valency
    #     results = check_valence(
    #         all_pos,
    #         all_numbers,
    #         valence,
    #         True,
    #         print_file,
    #         prog_str(work_str),
    #     )
    #     n_atoms_list = np.concatenate([n_atoms_list, np.ones(n_mols) * n_atoms])
    #     valid_mols = np.concatenate([valid_mols, results["valid_mol"]])
    #     valid_atoms = np.concatenate([valid_atoms, results["valid_atom"]])
    # valid_stats = pd.DataFrame({"n_atoms": n_atoms_list, "valid_mol": valid_mols, "valid_atoms": valid_atoms})
    # valid_stats["valid_atoms_frac"] = valid_stats["valid_atoms"] / valid_stats["n_atoms"]
    # with open('valency-results.pkl', 'wb') as f:
    #     pickle.dump(valid_stats, f)
