from typing import List

import logging
import os
import zipfile
import urllib
import numpy as np
import ase
import rdkit.Chem as Chem
from symphony.data.utils import download_url, extract_zip, molecule_sanity


QM9_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
)

def load_qm9(
    root_dir: str, use_edm_splits: bool, check_molecule_sanity: bool
) -> List[ase.Atoms]:
    """Load the QM9 dataset."""
    if use_edm_splits and check_molecule_sanity:
        raise ValueError(
            "EDM splits are not compatible with sanity checks. Set check_molecule_sanity as False."
        )

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    path = download_url(QM9_URL, root_dir)
    extract_zip(path, root_dir)

    raw_mols_path = os.path.join(root_dir, "gdb9.sdf")
    supplier = Chem.SDMolSupplier(raw_mols_path, removeHs=False, sanitize=False)

    mols_as_ase = []
    for mol in supplier:
        if mol is None:
            continue

        # Check that the molecule passes some basic checks from Posebusters.
        if check_molecule_sanity:
            sane = molecule_sanity(mol)
            if not sane:
                continue

        # Convert to ASE.
        mol_as_ase = ase.Atoms(
            numbers=np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
            positions=np.asarray(mol.GetConformer(0).GetPositions()),
        )
        mols_as_ase.append(mol_as_ase)

    if use_edm_splits:
        try:
            splits = np.load(os.path.join(root_dir, "edm_splits.npz"))
        except:
            splits = get_edm_splits(root_dir)
        mols_as_ase_train = [mols_as_ase[idx] for idx in splits["train"]]
        mols_as_ase_valid = [mols_as_ase[idx] for idx in splits["valid"]]
        mols_as_ase_test = [mols_as_ase[idx] for idx in splits["test"]]
        # Combine splits in order.
        mols_as_ase = [*mols_as_ase_train, *mols_as_ase_valid, *mols_as_ase_test]

    logging.info(f"Loaded {len(mols_as_ase)} molecules.")
    return mols_as_ase


def get_edm_splits(root_dir: str):
    """
    Adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/data/prepare/qm9.py.
    """

    def is_int(string):
        try:
            int(string)
            return True
        except:
            return False

    logging.info("Using EDM splits. This drops some molecules.")
    gdb9_url_excluded = "https://springernature.figshare.com/ndownloader/files/3195404"
    gdb9_txt_excluded = os.path.join(root_dir, "uncharacterized.txt")
    urllib.request.urlretrieve(gdb9_url_excluded, filename=gdb9_txt_excluded)

    # First, get list of excluded indices.
    excluded_strings = []
    with open(gdb9_txt_excluded) as f:
        lines = f.readlines()
        excluded_strings = [line.split()[0] for line in lines if len(line.split()) > 0]

    excluded_idxs = [int(idx) - 1 for idx in excluded_strings if is_int(idx)]

    assert (
        len(excluded_idxs) == 3054
    ), "There should be exactly 3054 excluded atoms. Found {}".format(
        len(excluded_idxs)
    )

    # Now, create a list of included indices.
    Ngdb9 = 133885
    Nexcluded = 3054

    included_idxs = np.array(sorted(list(set(range(Ngdb9)) - set(excluded_idxs))))

    # Now, generate random permutations to assign molecules to training/validation/test sets.
    Nmols = Ngdb9 - Nexcluded
    assert Nmols == len(
        included_idxs
    ), "Number of included molecules should be equal to Ngdb9 - Nexcluded. Found {} {}".format(
        Nmols, len(included_idxs)
    )

    Ntrain = 100000
    Ntest = int(0.1 * Nmols)
    Nvalid = Nmols - (Ntrain + Ntest)

    # Generate random permutation.
    np.random.seed(0)
    data_permutation = np.random.permutation(Nmols)

    train, valid, test, extra = np.split(
        data_permutation, [Ntrain, Ntrain + Nvalid, Ntrain + Nvalid + Ntest]
    )

    assert len(extra) == 0, "Split was inexact {} {} {} {}".format(
        len(train), len(valid), len(test), len(extra)
    )

    train = included_idxs[train]
    valid = included_idxs[valid]
    test = included_idxs[test]

    splits = {"train": train, "valid": valid, "test": test}
    np.savez(os.path.join(root_dir, "edm_splits.npz"), **splits)
    print(os.path.join(root_dir, "edm_splits.npz"))

    # Cleanup file.
    try:
        os.remove(gdb9_txt_excluded)
    except OSError:
        pass

    return splits
