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
from sh import gunzip
import numpy as np
import ase
import rdkit.Chem as Chem


from symphony.data import datasets
from symphony.models import ptable
from symphony import datatypes


TMQM_URL = "https://github.com/bbskjelstad/tmqm.git"


def _molecule_to_structure(molecule: ase.Atoms) -> datatypes.Structures:
    """Converts a molecule to a datatypes.Structures object."""
    return datatypes.Structures(
        nodes=datatypes.NodesInfo(
            positions=np.asarray(molecule.positions),
            species=molecule.numbers - 1
        ),
        edges=None,
        receivers=None,
        senders=None,
        globals=None,
        n_node=np.asarray([len(molecule.numbers)]),
        n_edge=None,
    )


class TMQMDataset(datasets.InMemoryDataset):
    """TMQM dataset."""

    def __init__(self, root_dir: str, num_train_molecules: int, 
                 num_val_molecules: int, num_test_molecules: int):
        super().__init__()
        
        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        logging.info("Using random splits.")
        if num_train_molecules is None or num_val_molecules is None or num_test_molecules is None:
            raise ValueError("num_train_molecules, num_val_molecules, and num_test_molecules must be provided.")
            
        self.root_dir = root_dir
        self.num_train_molecules = num_train_molecules
        self.num_val_molecules = num_val_molecules
        self.num_test_molecules = num_test_molecules

        self.molecules = load_tmqm(self.root_dir)
        logging.info("Loaded TMQM dataset.")

    def structures(self) -> Iterable[datatypes.Structures]:
        for molecule in self.molecules:
            yield _molecule_to_structure(molecule) 

    @staticmethod
    def species_to_atom_types() -> Dict[int, str]:
        return {
            i: ptable.symbols[i] for i in range(1, 81)
        }

    def split_indices(self) -> Dict[str, Set[int]]:
        # Create a random permutation of the indices.
        np.random.seed(0)
        indices = np.random.permutation(len(self.molecules))
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


def clone_url(url: str, root: str) -> str:
    """Clone if repo does not exist in root already. Returns path to repo."""
    repo_path = os.path.join(root, url.rpartition("/")[-1].rpartition(".")[0])

    if os.path.exists(repo_path):
        logging.info(f"Using cloned repo: {repo_path}")
        return repo_path

    logging.info(f"Cloning {url} to {repo_path}")
    Repo.clone_from(url, repo_path)

    return repo_path


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


def extract_tar(path: str, root: str):
    """Extract tar."""
    logging.info(f"Extracting {path} to {root}...")
    with tarfile.TarFile(path, "r") as f:
        f.extractall(path=root)


def load_tmqm(root_dir: str) -> List[ase.Atoms]:
    """Load the TMQM dataset."""
    mols_as_ase = []
    data_path = root_dir
    xyzs_path = os.path.join(data_path, "xyz")
    if os.path.exists(xyzs_path):
        logging.info(f"Using downloaded data: {xyzs_path}")
    else:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        logging.info(f"Cloning TMQM repository to {root_dir}...")
        _ = clone_url(TMQM_URL, root_dir)
        if not os.path.exists(xyzs_path):
            os.makedirs(xyzs_path)

        for i in range(1, 3):
            gz_path = os.path.join(data_path, "tmqm/data", f"tmQM_X{i}.xyz.gz")
            logging.info(f"Unzipping {gz_path}...")
            gunzip(gz_path)

            mol_file = os.path.join(data_path, "tmqm/data", f"tmQM_X{i}.xyz")
            with open(mol_file, "r") as f:
                all_xyzs = f.read().split("\n\n")
                for xyz_n, xyz in enumerate(all_xyzs):
                    if xyz == "":
                        continue
                    xyz_lines = xyz.split("\n")
                    assert len(xyz_lines) == int(xyz_lines[0]) + 2
                    with open(os.path.join(xyzs_path, f"X{i}_{xyz_n}.xyz"), "w") as f:
                        f.write(xyz)

    for mol_file in tqdm.tqdm(os.listdir(xyzs_path)):
        mol_as_ase = ase.io.read(os.path.join(xyzs_path, mol_file), format="xyz")
        if mol_as_ase is None:
            continue
        mols_as_ase.append(mol_as_ase)

    logging.info(f"Loaded {len(mols_as_ase)} molecules.")
    return mols_as_ase
