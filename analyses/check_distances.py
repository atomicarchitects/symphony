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


# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
    },
    "N": {
        "H": 101,
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
    },
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
    },
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121},
}

bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}


def get_parser():
    """Setup parser for command line arguments"""
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "mol_path",
        help="Path to generated molecule in .xyz format",
    )
    main_parser.add_argument(
        "--min_dist",
        default=1.0,
        type=float,
        help="minimum interatomic distance (default: %(default)s)",
    )

    return main_parser


def get_interatomic_distances(positions) -> np.ndarray:
    """Returns the interatomic distances of the given molecule."""
    distances = []
    for atom1, atom2 in itertools.combinations(positions, 2):
        distances.append(np.linalg.norm(atom1 - atom2))
    return np.array(distances)


def check_distances(positions, min_dist: 1, return_distances=False) -> bool:
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
    if return_distances:
        return np.all(distances > min_dist), distances
    return np.all(distances > min_dist)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    mol = ase.io.read(args.mol_path)

    valid, distances = check_distances(mol.positions, args.min_dist, return_distances=True)
    print(f'All distances are greater than {args.min_dist}: {valid}')
