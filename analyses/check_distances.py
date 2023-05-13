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


BOND_LENGTHS = [
    [0.7,  1.1,  1.0,  0.97, 0.91],
    [1.1,  1.2,  1.17, 1.1,  1.27],
    [1.0,  1.17, 1.09, 1.15, 1.3],
    [0.97, 1.1,  1.15, 1.2,  1.35],
    [0.91, 1.27, 1.3,  1.35, 1.4]
]


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
        default=0.9,
        type=float,
        help="minimum interatomic distance (default: %(default)s)",
    )
    main_parser.add_argument(
        "--max_dist",
        default=10.0,
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


def check_distances(positions, min_dist: 1, max_dist: 10) -> bool:
    """
    Checks if the molecule has any interatomic distances outside the given range.
    Args:
        positions (list of numpy.ndarray): list of positions of atoms in euclidean
            space (n_atoms x 3) for each molecule
        min_dist (float): minimum allowed interatomic distance (in angstroms)
        max_dist (float): maximum allowed interatomic distance (in angstroms)
    Returns:
        a list containing 
    """
    dist_valid = []
    for mol_positions in positions:
        distances = get_interatomic_distances(mol_positions)
        dist_valid.append(np.all(distances > min_dist) and np.all(distances < max_dist))
    return dist_valid


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
    max_dist = args.max_dist

    # print the chosen settings
    print(f"\nMinimum valid distance:\n{min_dist}\nMaximum distance:\n{max_dist}\n")

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
        valid_dists = np.concatenate([valid_dists, check_distances(all_pos, min_dist, max_dist)])
        n_atoms_list = np.concatenate([n_atoms_list, np.ones(n_mols) * n_atoms])

    valid_stats = pd.DataFrame({"n_atoms": n_atoms_list, "valid_distances": valid_dists})
    with open('distance-results.pkl', 'wb') as f:
        pickle.dump(valid_stats, f)
