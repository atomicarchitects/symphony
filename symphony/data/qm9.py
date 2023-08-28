from typing import List

import logging
import os
import zipfile
from urllib.request import urlopen
from urllib.error import URLError
import numpy as np
import ase
import rdkit.Chem as Chem

QM9_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
)


def download_url(url: str, root: str) -> str:
    """Download if file does not exist in root already. Returns path to file."""
    filename = url.rpartition("/")[2]
    file_path = os.path.join(root, filename)

    try:
        from tqdm import tqdm

        progress = True
    except ImportError:
        progress = False

    try:
        if os.path.exists(file_path):
            logging.info(f"Using downloaded file: {file_path}")
            return file_path
        data = urlopen(url)
    except URLError:
        # No internet connection
        if os.path.exists(file_path):
            logging.info(f"No internet connection! Using downloaded file: {file_path}")
            return file_path

        raise

    chunk_size = 1024
    total_size = int(data.info()["Content-Length"].strip())

    if os.path.exists(file_path):
        if os.path.getsize(file_path) == total_size:
            logging.info(f"Using downloaded and verified file: {file_path}")
            return file_path

    logging.info(f"Downloading {url} to {file_path}")

    with open(file_path, "wb") as f:
        if progress:
            with tqdm(total=total_size) as pbar:
                while True:
                    chunk = data.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(chunk_size)
        else:
            while True:
                chunk = data.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

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


def load_qm9(root_dir: str) -> List[ase.Atoms]:
    """Load QM9 dataset."""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    path = download_url(QM9_URL, root_dir)
    extract_zip(path, root_dir)

    supplier = Chem.SDMolSupplier(
        os.path.join(root_dir, "gdb9.sdf"), removeHs=False, sanitize=True
    )
    mols_as_ase = []
    for mol in supplier:
        if mol is None:
            continue

        mol_as_ase = ase.Atoms(
            numbers=np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
            positions=np.asarray(mol.GetConformer(0).GetPositions()),
        )
        mols_as_ase.append(mol_as_ase)

    return mols_as_ase
