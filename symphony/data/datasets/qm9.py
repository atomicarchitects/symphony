from typing import List, Iterable, Dict, Set

import tqdm
from absl import logging
import os
import zipfile
import urllib
import numpy as np
import ase
import rdkit.Chem as Chem


from symphony.data import datasets
from symphony import datatypes


QM9_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
)


def _molecule_to_structure(molecule: ase.Atoms) -> datatypes.Structures:
    """Converts a molecule to a datatypes.Structures object."""
    return datatypes.Structures(
        nodes=datatypes.NodesInfo(
            positions=np.asarray(molecule.positions),
            species=np.searchsorted(np.asarray([1, 6, 7, 8, 9]), molecule.numbers),
        ),
        edges=None,
        receivers=None,
        senders=None,
        globals=None,
        n_node=np.asarray([len(molecule.numbers)]),
        n_edge=None,
    )


class QM9Dataset(datasets.InMemoryDataset):
    """QM9 dataset."""

    def __init__(
        self,
        root_dir: str,
        check_molecule_sanity: bool,
        use_edm_splits: bool,
        num_train_molecules: int,
        num_val_molecules: int,
        num_test_molecules: int,
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        if use_edm_splits:
            logging.info("Using EDM splits for QM9.")
            if check_molecule_sanity:
                raise ValueError(
                    "EDM splits are not compatible with molecule sanity checks."
                )
        else:
            logging.info("Using random (non-EDM) splits.")

        self.root_dir = root_dir
        self.check_molecule_sanity = check_molecule_sanity
        self.use_edm_splits = use_edm_splits
        self.num_train_molecules = num_train_molecules
        self.num_val_molecules = num_val_molecules
        self.num_test_molecules = num_test_molecules
        self.molecules = None

    def structures(self) -> Iterable[datatypes.Structures]:
        if self.molecules is None:
            self.molecules = load_qm9(self.root_dir, self.check_molecule_sanity)

        for molecule in self.molecules:
            yield _molecule_to_structure(molecule)

    @staticmethod
    def species_to_atom_types() -> Dict[int, str]:
        return {
            0: "H",
            1: "C",
            2: "N",
            3: "O",
            4: "F",
        }

    def split_indices(self) -> Dict[str, Set[int]]:
        """Return a dictionary of indices for each split."""

        # If EDM splits are used, return the splits.
        if self.use_edm_splits:
            return get_edm_splits(
                self.root_dir,
                self.num_train_molecules,
                self.num_val_molecules,
                self.num_test_molecules,
            )

        # Create a random permutation of the indices.
        np.random.seed(0)
        indices = np.random.permutation(len(self.molecules))
        permuted_indices = {
            "train": indices[: self.num_train_molecules],
            "val": indices[
                self.num_train_molecules : self.num_train_molecules
                + self.num_val_molecules
            ],
            "test": indices[
                self.num_train_molecules
                + self.num_val_molecules : self.num_train_molecules
                + self.num_val_molecules
                + self.num_test_molecules
            ],
        }
        return permuted_indices


def download_url(url: str, root: str) -> str:
    """Download if file does not exist in root already. Returns path to file."""
    filename = url.rpartition("/")[2]
    file_path = os.path.join(root, filename)

    try:
        if os.path.exists(file_path):
            logging.info(f"Using downloaded file: {file_path}")
            return file_path
        data = urllib.request.urlopen(url)
    except urllib.error.URLError:
        # No internet connection
        if os.path.exists(file_path):
            logging.info(f"No internet connection! Using downloaded file: {file_path}")
            return file_path

        raise ValueError(f"Could not download {url}")

    chunk_size = 1024
    total_size = int(data.info()["Content-Length"].strip())

    if os.path.exists(file_path):
        if os.path.getsize(file_path) == total_size:
            logging.info(f"Using downloaded and verified file: {file_path}")
            return file_path

    logging.info(f"Downloading {url} to {file_path}")

    with open(file_path, "wb") as f:
        with tqdm.tqdm(total=total_size) as pbar:
            while True:
                chunk = data.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(chunk_size)

    return file_path


def extract_zip(path: str, root: str):
    """Extract zip if content does not exist in root already."""
    logging.info(f"Extracting {path} to {root}...")
    with zipfile.ZipFile(path, "r") as f:
        for name in f.namelist():
            if name.endswith("/"):
                logging.info(f"Skip directory {name}")
                continue
            out_path = os.path.join(root, name)
            file_size = f.getinfo(name).file_size
            if os.path.exists(out_path) and os.path.getsize(out_path) == file_size:
                logging.info(f"Skip existing file {name}")
                continue
            logging.info(f"Extracting {name} to {root}...")
            f.extract(name, root)


def molecule_sanity(mol: Chem.Mol) -> bool:
    """Check that the molecule passes some basic sanity checks from Posebusters.
    Source: https://github.com/maabuu/posebusters/blob/main/posebusters/modules/sanity.py
    """

    errors = Chem.rdmolops.DetectChemistryProblems(
        mol, sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_ALL
    )
    types = [error.GetType() for error in errors]
    num_frags = len(Chem.rdmolops.GetMolFrags(mol, asMols=False, sanitizeFrags=False))

    results = {
        "passes_valence_checks": "AtomValenceException" not in types,
        "passes_kekulization": "AtomKekulizeException" not in types,
        "passes_rdkit_sanity_checks": len(errors) == 0,
        "all_atoms_connected": num_frags <= 1,
    }
    return all(results.values())


def load_qm9(
    root_dir: str,
    check_molecule_sanity: bool = True,
) -> List[ase.Atoms]:
    """Load the QM9 dataset."""

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

    return mols_as_ase


def get_edm_splits(
    root_dir: str,
    num_train_molecules: int,
    num_val_molecules: int,
    num_test_molecules: int,
) -> Dict[str, np.ndarray]:
    """Adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/qm9/data/prepare/qm9.py."""

    def is_int(string: str) -> bool:
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

    # Now, generate random permutations to assign molecules to training/valation/test sets.
    Nmols = Ngdb9 - Nexcluded
    assert Nmols == len(
        included_idxs
    ), "Number of included molecules should be equal to Ngdb9 - Nexcluded. Found {} {}".format(
        Nmols, len(included_idxs)
    )

    Ntrain = 100000
    Ntest = int(0.1 * Nmols)
    Nval = Nmols - (Ntrain + Ntest)

    # Generate random permutation.
    np.random.seed(0)
    data_permutation = np.random.permutation(Nmols)

    train, val, test, extra = np.split(
        data_permutation, [Ntrain, Ntrain + Nval, Ntrain + Nval + Ntest]
    )

    assert len(extra) == 0, "Split was inexact {} {} {} {}".format(
        len(train), len(val), len(test), len(extra)
    )

    train = included_idxs[train]
    val = included_idxs[val]
    test = included_idxs[test]

    if num_train_molecules is not None:
        logging.info(f"Using {num_train_molecules} training molecules out of {len(train)} in EDM split.")
        train = train[:num_train_molecules]
    if num_val_molecules is not None:
        logging.info(f"Using {num_val_molecules} validation molecules out of {len(val)} in EDM split.")
        val = val[:num_val_molecules]
    if num_test_molecules is not None:
        logging.info(f"Using {num_test_molecules} test molecules out of {len(test)} in EDM split.")
        test = test[:num_test_molecules]

    splits = {"train": train, "val": val, "test": test}

    # Cleanup file.
    try:
        os.remove(gdb9_txt_excluded)
    except OSError:
        pass

    return splits
