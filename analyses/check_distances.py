##########################################################
# From G-SchNet repository                               #
# https://github.com/atomistic-machine-learning/G-SchNet #
##########################################################

import argparse
from typing import List
import ase
import itertools
import os
import pickle
import numpy as np
import pandas as pd
from schnetpack import Properties
from analyses.analysis import update_dict


# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121},
        }

bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}


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
    main_parser.add_argument(
        "--min_dist",
        default=1.0,
        type=float,
        help="minimum interatomic distance (default: %(default)s)",
    )
    main_parser.add_argument(
        "--print_file",
        help="Use to limit the printing if results are "
        "written to a file instead of the console ("
        "e.g. if running on a cluster)",
        action="store_true",
    )

    return main_parser


def get_interatomic_distances(positions) -> np.ndarray:
    """Returns the interatomic distances of the given molecule."""
    distances = []
    for atom1, atom2 in itertools.combinations(positions, 2):
        distances.append(np.linalg.norm(atom1 - atom2))
    return np.array(distances)


def check_distances(positions, min_dist: 1) -> bool:
    """
    Checks if the molecule has any interatomic distances less than the specified distance.
    Args:
        positions (numpy.ndarray): list of positions of atoms in euclidean
            space (n_atoms x 3)
        min_dist (float): minimum allowed interatomic distance (in angstroms)
    Returns:
        True if all interatomic distances are greater than the minimum distance, False otherwise
    """
    distances = get_interatomic_distances(positions)
    return np.all(distances > min_dist)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print_file = args.print_file

    # read input file or fuse dictionaries if mol_path is a folder
    if not os.path.isdir(args.mol_path):
        if not os.path.isfile(args.mol_path):
            print(
                f"\n\nThe specified data path ({args.mol_path}) is neither a file "
                f"nor a directory! Please specify a different data path."
            )
            raise FileNotFoundError
        else:
            with open(args.mol_path, "rb") as f:
                res = pickle.load(f)  # read input file
    else:
        print(f"\n\nFusing .mol_dict files in folder {args.mol_path}...")
        mol_files = [f for f in os.listdir(args.mol_path) if f.endswith(".mol_dict")]
        if len(mol_files) == 0:
            print(
                f"Could not find any .mol_dict files at {args.mol_path}! Please "
                f"specify a different data path!"
            )
            raise FileNotFoundError
        res = {}
        for file in mol_files:
            with open(os.path.join(args.mol_path, file), "rb") as f:
                cur_res = pickle.load(f)
                update_dict(res, cur_res)
        res = dict(sorted(res.items()))  # sort dictionary keys
        print(f"...done!")

    # get distance bounds
    min_dist = args.min_dist
    # print the chosen settings
    print(f"\nMinimum valid distance:\n{min_dist}\n")

    n_atoms_list = np.array([], dtype=np.int32)
    valid_dists = np.array([])
    valid_atoms = np.array([])

    # Check distances
    for n_atoms in res:
        if not isinstance(n_atoms, int) or n_atoms == 0:
            continue

        prog_str = lambda x: f"Checking {x} for molecules of length {n_atoms}"
        work_str = "valence"
        if not print_file:
            print("\033[K", end="\r", flush=True)
            print(prog_str(work_str) + " (0.00%)", end="\r", flush=True)
        else:
            print(prog_str(work_str), flush=True)

        d = res[n_atoms]
        all_pos = d[Properties.R]
        all_numbers = d[Properties.Z]
        n_mols = len(all_pos)

        # check distances
        valid_dists = np.concatenate([valid_dists, check_distances(all_pos, min_dist)])
        n_atoms_list = np.concatenate([n_atoms_list, np.ones(n_mols) * n_atoms])

    valid_stats = pd.DataFrame({"n_atoms": n_atoms_list, "valid_distances": valid_dists})
    with open('distance-results.pkl', 'wb') as f:
        pickle.dump(valid_stats, f)
