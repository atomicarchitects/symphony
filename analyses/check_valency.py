##########################################################
# From G-SchNet repository                               #
# https://github.com/atomistic-machine-learning/G-SchNet #
##########################################################

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from schnetpack import Properties

from analyses.analysis import update_dict
from analyses.utility_classes import Molecule


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
        "--valence",
        default=[1, 1, 6, 4, 7, 3, 8, 2, 9, 1],
        type=int,
        nargs="+",
        help="the valence of atom types in the form "
        "[type1 valence type2 valence ...] "
        "(default: %(default)s)",
    )
    main_parser.add_argument(
        "--print_file",
        help="Use to limit the printing if results are "
        "written to a file instead of the console ("
        "e.g. if running on a cluster)",
        action="store_true",
    )

    return main_parser


def check_valency(
    positions,
    numbers,
    valence,
    filter_by_valency=True,
    print_file=True,
    prog_str=None,
    picklable_mols=False,
):
    """
    Build utility_classes.Molecule objects from provided atom positions and types
    of a set of molecules and assess whether they are meeting the valency
    constraints or not (i.e. all of their atoms have the correct number of bonds).
    Note that all input molecules need to have the same number of atoms.

    Args:
        positions (list of numpy.ndarray): list of positions of atoms in euclidean
            space (n_atoms x 3) for each molecule
        numbers (numpy.ndarray): list of nuclear charges/types of atoms
            (e.g. 1 for hydrogens, 6 for carbons etc.) for each molecule
        valence (numpy.ndarray): list of valency of each atom type where the index in
            the list corresponds to the type (e.g. [0, 1, 0, 0, 0, 0, 4, 3, 2, 1] for
            qm9 molecules as H=type 1 has valency of 1, C=type 6 has valency of 4,
            N=type 7 has valency of 3 etc.)
        filter_by_valency (bool, optional): whether molecules that fail the valency
            check should be marked as invalid, else all input molecules will be
            classified as valid but the connectivity matrix is still computed and
            returned (default: True)
        print_file (bool, optional): set True to suppress printing of progress string
            (default: True)
        prog_str (str, optional): specify a custom progress string (default: None)
        picklable_mols (bool, optional): set True to remove all the information in
            the returned list of utility_classes.Molecule objects that can not be
            serialized with pickle (e.g. the underlying Open Babel ob.Mol object,
            default: False)

    Returns:
        dict (str->list/numpy.ndarray): a dictionary containing a list of
            utility_classes.Molecule ojbects under the key 'mols', a numpy.ndarray with
            the corresponding (n_atoms x n_atoms) connectivity matrices under the key
            'connectivity', a numpy.ndarray (key 'valid_mol') that marks whether a
            molecule has passed (entry=1) or failed (entry=0) the valency check if
            filter_by_valency is True (otherwise it will be 1 everywhere), and a numpy.ndarray
            (key 'valid_atom') that marks the number of atoms in each molecule that pass
            the valency check if filter_by_valency is True (otherwise it will be n_atoms everywhere)
    """
    n_atoms = len(numbers[0])
    n_mols = len(numbers)
    thresh = n_mols if n_mols < 30 else 30
    connectivity = np.zeros((len(positions), n_atoms, n_atoms))
    valid_mol = np.ones(len(positions), dtype=bool)
    valid_atom = np.ones(len(positions)) * n_atoms
    mols = []
    for i, (pos, num) in enumerate(zip(positions, numbers)):
        mol = Molecule(pos, num, store_positions=False)
        con_mat = mol.get_connectivity()
        random_ord = range(len(pos))
        # filter incorrect valence if desired
        if filter_by_valency:
            nums = num
            # try to fix connectivity if it isn't correct already
            for _ in range(10):
                if np.all(np.sum(con_mat, axis=0) == valence[nums]):
                    val = True
                    break
                else:
                    val = False
                    con_mat = mol.get_fixed_connectivity()
                    if np.all(np.sum(con_mat, axis=0) == valence[nums]):
                        val = True
                        break
                    random_ord = np.random.permutation(range(len(pos)))
                    mol = Molecule(pos[random_ord], num[random_ord])
                    con_mat = mol.get_connectivity()
                    nums = num[random_ord]
            valid_mol[i] = val
            valid_atom[i] = np.sum(np.equal(np.sum(con_mat, axis=0), valence[nums]))

            if ((i + 1) % thresh == 0) and not print_file and prog_str is not None:
                print("\033[K", end="\r", flush=True)
                print(
                    f"{prog_str} ({100 * (i + 1) / n_mols:.2f}%)", end="\r", flush=True
                )

        # reverse random order and save fixed connectivity matrix
        rand_ord_rev = np.argsort(random_ord)
        connectivity[i] = con_mat[rand_ord_rev][:, rand_ord_rev]
        if picklable_mols:
            mol.get_fp_bits()
            mol.get_can()
            mol.get_mirror_can()
            mol.remove_unpicklable_attributes(restorable=False)
        mols += [mol]
    return {
        "mols": mols,
        "connectivity": connectivity,
        "valid_mol": valid_mol,
        "valid_atom": valid_atom,
    }


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

    # compute array with valence of provided atom types
    max_type = max(args.valence[::2])
    valence = np.zeros(max_type + 1, dtype=int)
    valence[args.valence[::2]] = args.valence[1::2]

    # print the chosen settings
    valence_str = ""
    for i in range(max_type + 1):
        if valence[i] > 0:
            valence_str += f"type {i}: {valence[i]}, "
    print(f"\nTarget valence:\n{valence_str[:-2]}\n")

    n_atoms_list = np.array([], dtype=np.int32)
    valid_mols = np.array([])
    valid_atoms = np.array([])

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

        # check valency
        results = check_valency(
            all_pos,
            all_numbers,
            valence,
            True,
            print_file,
            prog_str(work_str),
        )
        n_atoms_list = np.concatenate([n_atoms_list, np.ones(n_mols) * n_atoms])
        valid_mols = np.concatenate([valid_mols, results["valid_mol"]])
        valid_atoms = np.concatenate([valid_atoms, results["valid_atom"]])
    valid_stats = pd.DataFrame({"n_atoms": n_atoms_list, "valid_mol": valid_mols, "valid_atoms": valid_atoms})
    valid_stats["valid_atoms_frac"] = valid_stats["valid_atoms"] / valid_stats["n_atoms"]
    with open('valency-results.pkl', 'wb') as f:
        pickle.dump(valid_stats, f)
