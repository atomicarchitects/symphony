##########################################################
# From G-SchNet repository                               #
# https://github.com/atomistic-machine-learning/G-SchNet #
##########################################################

import argparse
import logging
import os
import pickle
import time
import sys

from ase.db import connect
import numpy as np
import pandas as pd
import tqdm
import yaml

sys.path.append("..")

from analyses import analysis
from analyses.check_valence import check_valence
from analyses.utility_functions import _get_atoms_per_type_str, _update_dict, fingerprints_similar, get_fingerprint


def get_parser():
    """Setup parser for command line arguments"""
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "mol_path",
        help="Path to generated molecules as an ASE db, "
        'computed statistics ("generated_molecules_statistics.pkl") will be '
        "stored in the same directory as the input file/s ",
    )
    main_parser.add_argument(
        "--data_path",
        help="Path to training data base (if provided, "
        "generated molecules can be compared/matched with "
        "those in the training data set)",
        default=None,
    )
    main_parser.add_argument(
        "--model_path",
        help="Path of directory containing the model that " "generated the molecules. ",
        default=None,
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
    main_parser.add_argument(
        "--init",
        type=str,
        default="C",
        help="An initial molecular fragment to start the generation process from.",
    )

    return main_parser


def find_duplicates(mols, valid=None, use_bits=False):
    """
    Identify duplicate molecules among a large amount of generated structures.
    The first found structure of each kind is kept as valid original and all following
    duplicating structures are marked as invalid (the molecular fingerprint and
    canonical smiles representation is used which means that different spatial
    conformers of the same molecular graph cannot be distinguished).

    Args:
        mols (list of ase.Atoms): list of all generated molecules
        valid (numpy.ndarray, optional): array of the same length as mols which flags
            molecules as valid (invalid molecules are not considered in the comparison
            process), if None, all molecules in mols are considered as valid (default:
            None)
        use_bits (bool, optional): set True to use the list of non-zero bits instead of
            the pybel.Fingerprint object when comparing molecules (results are
            identical, default: False)

    Returns:
        duplicating (numpy.ndarray): array of length n_mols where entry i is -1 if molecule i is
            an original structure (not a duplicate) and otherwise it is the index j of
            the original structure that molecule i duplicates (j<i)
        duplicating_count (numpy.ndarray): array of length n_mols that is 0 for all duplicates and the
            number of identified duplicates for all original structures (therefore
            the sum over this array is the total number of identified duplicates)
    """
    if valid is None:
        valid = np.ones(len(mols), dtype=bool)
    else:
        valid = valid.copy()
    accepted_dict = {}
    duplicating = -np.ones(len(mols), dtype=int)
    duplicate_count = np.zeros(len(mols), dtype=int)
    for i, mol1 in enumerate(mols):
        if not valid[i]:
            continue
        mol_key = _get_atoms_per_type_str(mol1)
        found = False
        if mol_key in accepted_dict:
            for j, mol2 in accepted_dict[mol_key]:
                # compare fingerprints
                fp2, symbols2 = get_fingerprint(mol2, use_bits=use_bits)
                if fingerprints_similar(mol1, fp2, symbols2, use_bits=use_bits):
                    found = True
                    valid[i] = False
                    duplicating[i] = j
                    duplicate_count[j] += 1
                    break
        if not found:
            accepted_dict = _update_dict(accepted_dict, key=mol_key, val=(i, mol1))
    return duplicating, duplicate_count


def find_in_training_data(
    mols, stats, stat_heads, model_path, data_path, print_file=False
):
    """
    Check whether generated molecules correspond to structures in the training database
    used for either training, validation, or as test data and update statistics array of
    generated molecules accordingly.

    Args:
        mols (list of ase.Atoms): generated molecules
        stats (numpy.ndarray): statistics of all generated molecules where rows
            correspond to molecules and columns correspond to available statistics
            (n_molecules x n_statistics)
        stat_heads (list of str): the names of the statistics stored in each row in
            stats (e.g. 'F' for the number of fluorine atoms or 'R5' for the number of
            rings of size 5)
        model_path (str): path to the folder containing the trained model used to
            generate the molecules
        data_path (str): full path to the training database
        print_file (bool, optional): set True to limit printing (e.g. if it is
            redirected to a file instead of displayed in a terminal, default: False)

    Returns:
        numpy.ndarray: updated statistics of all generated molecules (stats['known']
        is 0 if a generated molecule does not correspond to a structure in the
        training database, it is 1 if it corresponds to a training structure,
        2 if it corresponds to a validation structure, and 3 if it corresponds to a
        test structure, stats['equals'] is -1 if stats['known'] is 0 and otherwise
        holds the index of the corresponding training/validation/test structure in
        the database at data_path)
    """
    print(f"\n\n2. Checking which molecules are new...")
    idx_known = stat_heads.index("known")

    # load training data
    dbpath = data_path
    if not os.path.isfile(dbpath):
        print(
            f"The provided training data base {dbpath} is no file, please specify "
            f"the correct path (including the filename and extension)!"
        )
        raise FileNotFoundError
    print(f"Using data base at {dbpath}...")

    if not os.path.exists(model_path):
        raise FileNotFoundError
    
    # Load config.
    saved_config_path = os.path.join(model_path, "config.yml")
    if not os.path.exists(saved_config_path):
        raise FileNotFoundError(f"No saved config found at {model_path}")

    logging.info("Saved config found at %s", saved_config_path)
    with open(saved_config_path, "r") as config_file:
        config = yaml.unsafe_load(config_file)

    train_idx = np.array(range(config.train_molecules[0], config.train_molecules[1]))
    val_idx = np.array(range(config.val_molecules[0], config.val_molecules[1]))
    test_idx = np.array(range(config.test_molecules[0], config.test_molecules[1]))
    all_idx = np.append(train_idx, val_idx)
    all_idx = np.append(all_idx, test_idx)

    print("\nComputing fingerprints of training data...")
    start_time = time.time()

    train_fps_dict = _get_training_fingerprints(
        dbpath, all_idx, print_file
    )

    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(
        f'...{len(all_idx)} fingerprints computed '
        f"in {h:d}h{m:02d}m{s:02d}s!"
    )

    print("\nComparing fingerprints...")
    start_time = time.time()
    stats = _compare_training_fingerprints(
        mols,
        train_fps_dict,
        all_idx,
        [len(val_idx), len(test_idx)],
        stats,
        stat_heads,
        print_file,
    )
    stats = stats.T
    stats[idx_known] = stats[idx_known]
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(f"... needed {h:d}h{m:02d}m{s:02d}s.")
    print(
        f"Number of new molecules: "
        f"{sum(stats[idx_known] == 0)+sum(stats[idx_known] == 3)}"
    )
    print(
        f"Number of molecules matching training data: " f"{sum(stats[idx_known] == 1)}"
    )
    print(
        f"Number of molecules matching validation data: "
        f"{sum(stats[idx_known] == 2)}"
    )
    print(f"Number of molecules matching test data: " f"{sum(stats[idx_known] == 3)}")

    return stats.T


def _get_training_fingerprints(
    dbpath, train_idx, print_file=True, use_bits=False
):
    """
    Get the fingerprints (FP2 from Open Babel), canonical smiles representation,
    and atoms per type string of all molecules in the training database.

    Args:
        dbpath (str): path to the training database
        train_idx (list of int): list containing the indices of training, validation,
            and test molecules in the database (it is assumed
            that train_idx[0:n_train] corresponds to training data,
            train_idx[n_train:n_train+n_validation] corresponds to validation data,
            and train_idx[n_train+n_validation:] corresponds to test data)
        print_file (bool, optional): set True to suppress printing of progress string
            (default: True)
        use_bits (bool, optional): set True to return the non-zero bits in the
            fingerprint instead of the pybel.Fingerprint object (default: False)

    Returns:
        dict (str->list of tuple): dictionary with the atoms per type string of each molecule
            as the keys, and 
    """
    train_fps = []
    with connect(dbpath) as conn:
        if not print_file:
            print("0.00%", end="\r", flush=True)
        for i, idx in enumerate(train_idx):
            idx = int(idx)
            try:
                row = conn.get(idx + 1)
            except:
                print(f"error getting idx={idx}")
            at = row.toatoms()
            train_fps += [get_fingerprint(at, use_bits)]
            if (i % 100 == 0 or i + 1 == len(train_idx)):
                print("\033[K", end="\r", flush=True)
                print(f"{100 * (i + 1) / len(train_idx):.2f}%", end="\r", flush=True)
    
    fp_dict = {}
    for i, fp in enumerate(train_fps):
        fp_dict = _update_dict(fp_dict, key=fp[-1], val=fp[:-1] + (i,))
    return fp_dict


def _compare_training_fingerprints(
    mols,
    train_fps,
    train_idx,
    thresh,
    stats,
    stat_heads,
    print_file=True,
    use_bits=False,
    max_heavy_atoms=9,
):
    """
    Compare fingerprints of generated and training data molecules to update the
    statistics of the generated molecules (to which training/validation/test
    molecule it corresponds, if any).

    Args:
        mols (list of ase.Atoms): generated molecules
        train_fps (dict (str->list of tuple)): dictionary with fingerprints of
            training/validation/test data as returned by _get_training_fingerprints_dict
        train_idx (list of int): list that maps the index of fingerprints in the
            train_fps dict to indices of the underlying training database (it is assumed
            that train_idx[0:n_train] corresponds to training data,
            train_idx[n_train:n_train+n_validation] corresponds to validation data,
            and train_idx[n_train+n_validation:] corresponds to test data)
        thresh (tuple of int): tuple containing the number of validation and test
            data molecules (n_validation, n_test)
        stats (numpy.ndarray): statistics of all generated molecules where rows
            correspond to molecules and columns correspond to available statistics
            (n_molecules x n_statistics)
        stat_heads (list of str): the names of the statistics stored in each row in
            stats (e.g. 'F' for the number of fluorine atoms or 'R5' for the number of
            rings of size 5)
        print_file (bool, optional): set True to limit printing (e.g. if it is
            redirected to a file instead of displayed in a terminal, default: True)
        use_bits (bool, optional): set True if the fingerprint is provided as a list of
            non-zero bits instead of the pybel.Fingerprint object (default: False)
        max_heavy_atoms (int, optional): the maximum number of heavy atoms in the
            training data set (i.e. 9 for qm9, default: 9)

    Returns:
        stats (numpy.ndarray): updated statistics
    """
    idx_known = stat_heads.index("known")
    idx_equals = stat_heads.index("equals")
    idx_val = stat_heads.index("valid_mol")
    n_val_mols, n_test_mols = thresh
    # get indices of valid molecules
    idcs = np.where(stats[:, idx_val])[0]
    if not print_file:
        print(f"0.00%", end="", flush=True)
    for i, idx in enumerate(idcs):
        mol = mols[idx]
        mol_key = _get_atoms_per_type_str(mol)
        # for now the molecule is considered to be new
        stats[idx, idx_known] = 0
        if np.sum(mol.numbers != 1) > max_heavy_atoms:
            continue  # cannot be in dataset
        if mol_key not in train_fps:
            continue
        for fp_train, symbols_train in train_fps[mol_key]:
            # compare fingerprints
            if fingerprints_similar(mol, fp_train, symbols_train, use_bits):
                # store index of match
                j = fp_train[-1]
                stats[idx, idx_equals] = train_idx[j]
                if j >= len(train_idx) - np.sum(thresh):
                    if j > len(train_idx) - n_test_mols:
                        stats[idx, idx_known] = 3  # equals test data
                    else:
                        stats[idx, idx_known] = 2  # equals validation data
                else:
                    stats[idx, idx_known] = 1  # equals training data
                break
        if not print_file:
            print("\033[K", end="\r", flush=True)
            print(f"{100 * (i + 1) / len(idcs):.2f}%", end="\r", flush=True)
    if not print_file:
        print("\033[K", end="", flush=True)
    return stats


def get_bond_stats(mol):
        """
        Retrieve the bond and ring count of the molecule. The bond count is
        calculated for every pair of types (e.g. C1N are all single bonds between
        carbon and nitrogen atoms in the molecule, C2N are all double bonds between
        such atoms etc.). The ring count is provided for rings from size 3 to 8 (R3,
        R4, ..., R8) and for rings greater than size eight (R>8).

        Args:
            mol (ase.Atoms): molecule

        Returns:
            dict (str->int): bond and ring counts
        """
        # 1st analyze bonds
        bond_stats = {}
        obmol = analysis.construct_obmol(mol)
        for bond_idx in range(obmol.NumBonds()):
            bond = obmol.GetBond(bond_idx)
            atom1 = bond.GetBeginAtom().GetAtomicNum()
            atom2 = bond.GetEndAtom().GetAtomicNum()
            type1 = analysis.NUMBER_TO_SYMBOL[min(atom1, atom2)]
            type2 = analysis.NUMBER_TO_SYMBOL[max(atom1, atom2)]
            id = f'{type1}{bond.GetBondOrder()}{type2}'
            bond_stats[id] = bond_stats.get(id, 0) + 1
        # remove twice counted bonds
        for bond_type in bond_stats.keys():
            if bond_type[0] == bond_type[2]:
                bond_stats[id] = int(bond_stats[id] / 2)

        # 2nd analyze rings
        rings = obmol.GetSSSR()
        if len(rings) > 0:
            for ring in rings:
                ring_size = ring.Size()
                if ring_size < 9:
                    bond_stats[f"R{ring_size}"] = bond_stats.get(f"R{ring_size}", 0) + 1
                else:
                    bond_stats["R>8"] = bond_stats.get("R>8", 0) + 1

        return bond_stats


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print_file = args.print_file

    molecules = []

    mol_path = args.mol_path
    if os.path.isdir(args.mol_path):
        mol_path = os.path.join(args.mol_path, f'generated_molecules_init={args.init}.db')
    if not os.path.isfile(mol_path):
        print(
            f"\n\nThe specified data path ({mol_path}) is neither a file "
            f"nor a directory! Please specify a different data path."
        )
        raise FileNotFoundError
    else:
        with connect(mol_path) as conn:
            for row in conn.select():
                molecules.append(row.toatoms())

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

    # initial setup of array for statistics and some counters
    n_generated = len(molecules)
    atom_cols = [
        "H",
        "C",
        "N",
        "O",
        "F",
    ]
    ring_bond_cols = [
        "H1C",
        "H1N",
        "H1O",
        "C1C",
        "C2C",
        "C3C",
        "C1N",
        "C2N",
        "C3N",
        "C1O",
        "C2O",
        "C1F",
        "N1N",
        "N2N",
        "N1O",
        "N2O",
        "N1F",
        "O1O",
        "O1F",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R>8",
    ]
    stat_heads = [
        "n_atoms",
        "valid_mol",
        "valid_atoms",
        "duplicating",
        "n_duplicates",
        "known",
        "equals",
        *atom_cols,
        *ring_bond_cols
    ]
    stats = np.empty((len(stat_heads), 0))
    valid = []  # True if molecule is valid w.r.t valence, False otherwise
    formulas = []

    start_time = time.time()
    for mol in tqdm.tqdm(molecules):
        n_atoms = len(mol.positions)

        # check valency
        valid_mol, valid_atoms = check_valence(mol, valence,)

        # collect statistics of generated data
        n_of_types = [np.sum(mol.numbers == i) for i in [6, 7, 8, 9, 1]]
        bond_stats = get_bond_stats(mol)
        stats_new = np.stack(
            (
                n_atoms,  # n_atoms
                valid_mol,  # valid molecules
                valid_atoms,  # valid atoms (atoms with correct valence)
                0,  # duplicating
                0,  # n_duplicates
                0,  # known
                0,  # equals
                *n_of_types,  # n_atoms per type
                *[bond_stats.get(k, 0) for k in ring_bond_cols],  # bond and ring counts
            ),
            axis=0,
        )
        stats_new = stats_new.reshape(stats_new.shape[0], 1)
        stats = np.hstack((stats, stats_new))

        valid.append(valid_mol)
        formulas.append(str(mol.symbols))

    duplicating, duplicate_count = find_duplicates(
        molecules, valid=valid, use_bits=False
    )

    stats[stat_heads.index("duplicating")] = np.array(duplicating)
    stats[stat_heads.index("n_duplicates")] = np.array(duplicate_count)

    if not print_file:
        print("\033[K", end="\r", flush=True)
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    h, m, s = int(h), int(m), int(s)
    print(f"Needed {h:d}h{m:02d}m{s:02d}s.")

    print(
        f"Number of generated molecules: {n_generated}\n"
        f"Number of duplicate molecules: {sum(duplicate_count)}"
    )

    n_valid_mol = 0
    for i in range(n_generated):
        if stats[2, i] == 1 and duplicating[i] == -1:
            n_valid_mol += 1

    print(f"Number of unique and valid molecules: {n_valid_mol}")

    # filter molecules which were seen during training
    if args.model_path is not None:
        stats = find_in_training_data(
            molecules,
            stats.T,
            stat_heads,
            args.model_path,
            args.data_path,
            print_file=print_file,
        )

    # store gathered statistics in metrics dataframe
    stats_df = pd.DataFrame(
        stats, columns=np.array(stat_heads)
    )
    stats_df.insert(0, "formula", formulas)
    metric_df_dict = analysis.get_results_as_dataframe(
        [""],
        ["total_loss", "atom_type_loss", "position_loss"],
        args.model_path,
    )
    cum_stats = {
        "valid_mol": stats_df["valid_mol"].sum() / len(stats_df),
        "valid_atoms": stats_df["valid_atoms"].sum() / stats_df["n_atoms"].sum(),
        "n_duplicates": stats_df["duplicating"].apply(lambda x: x != -1).sum(),
        "known": stats_df["known"].apply(lambda x: x > 0).sum(),
        "known_train": stats_df["known"].apply(lambda x: x == 1).sum(),
        "known_val": stats_df["known"].apply(lambda x: x == 2).sum(),
        "known_test": stats_df["known"].apply(lambda x: x == 3).sum(),
    }
    
    for col_name in ring_bond_cols:
        cum_stats[col_name] = stats_df[col_name].sum()

    cum_stats_df = pd.DataFrame(
        cum_stats, columns=list(cum_stats.keys()), index=[0]
    )

    metric_df_dict["generated_stats_overall"] = cum_stats_df
    metric_df_dict["generated_stats"] = stats_df

    # store results in pickle file
    stats_path = os.path.join(args.mol_path, f"generated_molecules_init={args.init}_statistics.pkl")
    if os.path.isfile(stats_path):
        file_name, _ = os.path.splitext(stats_path)
        expand = 0
        while True:
            expand += 1
            new_file_name = file_name + "_" + str(expand)
            if os.path.isfile(new_file_name + ".pkl"):
                continue
            else:
                stats_path = new_file_name + ".pkl"
                break
    with open(stats_path, "wb") as f:
        pickle.dump(metric_df_dict, f)
