from typing import List, Iterable, Dict, Set

import tqdm
from absl import logging
import os
import zipfile
import tarfile
import urllib
import urllib.error
import urllib.request
from git import Repo
import numpy as np
import pandas as pd
import ase
import pickle


from symphony.data import datasets
from symphony.models import ptable
from symphony import datatypes


PEROV_URL = "https://github.com/txie-93/cdvae.git"


def _molecule_to_structure(molecule: ase.Atoms, supercell: bool) -> datatypes.Structures:
    """Converts a molecule to a datatypes.Structures object."""
    if supercell:
        # make supercell if structure is too small
        num_atoms = molecule.numbers.shape[0]
        if num_atoms < 30:
            if num_atoms >= 30 / 2:
                P = np.eye(3)
                j = np.random.choice(np.arange(3))
                P[j, j] = 2
            elif num_atoms >= 30 / 4:
                P = 2 * np.eye(3)
                j = np.random.choice(np.arange(3))
                P[j, j] = 1
            else:
                P = 2 * np.eye(3)
            molecule = ase.build.make_supercell(molecule, P)
    return datatypes.Structures(
        nodes=datatypes.NodesInfo(
            positions=np.asarray(molecule.positions),
            species=molecule.numbers - 1
        ),
        edges=None,
        receivers=None,
        senders=None,
        globals=datatypes.GlobalsInfo(
            cell=np.asarray(molecule.cell),
        ),
        n_node=np.asarray([len(molecule.numbers)]),
        n_edge=None,
    )


class PerovDataset(datasets.InMemoryDataset):
    """TMQM dataset."""

    def __init__(self, root_dir: str, supercell: bool):
        super().__init__()
        
        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir

        splits = load_perov(self.root_dir, supercell)  # here: {split: [list of molecules]}
        logging.info("Loaded perov dataset.")

        self.num_train_molecules = len(splits['train'])
        self.num_val_molecules = len(splits['val'])
        self.num_test_molecules = len(splits['test'])
        self.molecules = splits['train'] + splits['val'] + splits['test']

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        #return np.array([
        #    3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 19, 20,
        #    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
        #    32, 33, 37, 38, 39, 40, 41, 42, 44, 45, 46, 
        #    47, 48, 49, 50, 51, 52, 55, 56, 57, 72, 73, 
        #    74, 75, 76, 77, 78, 79, 80, 81, 82, 83
        #])
        return np.arange(1, 84)

    def structures(self) -> Iterable[datatypes.Structures]:
        return self.molecules 

    @staticmethod
    def species_to_atom_types() -> Dict[int, str]:
        return {
            i: ptable.symbols[i] for i in PerovDataset.get_atomic_numbers()
        }

    def split_indices(self) -> Dict[str, Set[int]]:
        # # Create a random permutation of the indices.
        # np.random.seed(0)
        # indices = np.random.permutation(len(self.molecules))
        indices = np.arange(len(self.molecules))
        permuted_indices = {
            "train": indices[:self.num_train_molecules],
            "val": indices[self.num_train_molecules:self.num_train_molecules + self.num_val_molecules],
            "test": indices[self.num_train_molecules + self.num_val_molecules:self.num_train_molecules + self.num_val_molecules + self.num_test_molecules],
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


def clone_url(url: str, root: str) -> str:
    """Clone if repo does not exist in root already. Returns path to repo."""
    repo_path = os.path.join(root, url.rpartition("/")[-1].rpartition(".")[0])

    if os.path.exists(repo_path):
        logging.info(f"Using cloned repo: {repo_path}")
        return repo_path

    logging.info(f"Cloning {url} to {repo_path}")
    Repo.clone_from(url, repo_path)

    return repo_path


def load_perov(root_dir: str, supercell: bool) -> List[ase.Atoms]:
    """Load the perov dataset."""
    dataset_splits = {}
    data_path = root_dir
    repo_path = os.path.join(data_path, "cdvae")
    cif_path = os.path.join(data_path, "cdvae", "data", "perov_5", "cif")
    mol_path = os.path.join(data_path, "cdvae", "data", "perov_5", "molecules.pkl")

    if os.path.exists(mol_path):
        logging.info(f"Using saved molecules: {mol_path}")
        with open(mol_path, "rb") as f:
            mols = pickle.load(f)
        return mols

    splits = ['train', 'val', 'test']

    if os.path.exists(cif_path):
        logging.info(f"Using downloaded data: {cif_path}")
    else:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        logging.info(f"Cloning perov repository to {root_dir}...")
        _ = clone_url(PEROV_URL, root_dir)
        os.makedirs(cif_path)

        for split in splits:
            csv_path = os.path.join(data_path, "cdvae", "data", "perov_5", f"{split}.csv")
            df = pd.read_csv(csv_path)
            os.makedirs(os.path.join(cif_path, split))
            for i in range(len(df)):
                cif_file = os.path.join(cif_path, split, f"{i}.cif")
                with open(cif_file, "w") as f:
                    f.write(df["cif"][i])

    for split in splits:
        mols = []
        for mol_file in tqdm.tqdm(os.listdir(os.path.join(cif_path, split))):
            mol_as_ase = ase.io.read(os.path.join(cif_path, split, mol_file), format="cif")
            if mol_as_ase is None:
                continue
            mols.append(_molecule_to_structure(mol_as_ase, supercell))
        dataset_splits[split] = mols

    if not os.path.exists(mol_path):
        logging.info(f"Saving molecules to {mol_path}...")
        with open(mol_path, "wb") as f:
            pickle.dump(dataset_splits, f)

    logging.info(f"Loaded {sum([len(v) for _, v in dataset_splits.items()])} molecules.")
    return dataset_splits

