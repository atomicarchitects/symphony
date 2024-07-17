"""Utilities for downloading and extracting datasets."""

from typing import Dict
import os

import jax.numpy as jnp
from absl import logging
import tqdm
from git import Repo
import zipfile
import tarfile
import urllib
import ml_collections

from symphony.data.datasets import dataset, platonic_solids, qm9, qm9_single, geom_drugs, tmqm


def get_atomic_numbers(dataset: str) -> Dict[str, int]:
    """Returns a dictionary mapping atomic symbols to atomic numbers."""
    if dataset == "qm9":
        return qm9.QM9Dataset.get_atomic_numbers()
    elif dataset == "tmqm":
        return tmqm.TMQMDataset.get_atomic_numbers()
    elif dataset == "platonic_solids":
        return platonic_solids.PlatonicSolidsDataset.get_atomic_numbers()
    elif dataset == "geom_drugs":
        return geom_drugs.GEOMDrugsDataset.get_atomic_numbers()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def species_to_atomic_numbers(
    species: jnp.ndarray, dataset: str,
) -> jnp.ndarray:
    """Returns the atomic numbers for the species."""
    atomic_numbers = get_atomic_numbers(dataset)
    return jnp.asarray(atomic_numbers)[species]
    

def get_dataset(config: ml_collections.ConfigDict) -> dataset.InMemoryDataset:
    """Creates the dataset of structures, as specified in the config."""

    if config.dataset == "qm9":
        return qm9.QM9Dataset(
            root_dir=config.root_dir,
            check_molecule_sanity=config.get("check_molecule_sanity", False),
            use_edm_splits=config.use_edm_splits,
            num_train_molecules=config.num_train_molecules,
            num_val_molecules=config.num_val_molecules,
            num_test_molecules=config.num_test_molecules,
        )
    
    elif config.dataset == "qm9_single":
        return qm9_single.QM9Dataset(
            root_dir=config.root_dir,
            check_molecule_sanity=config.get("check_molecule_sanity", False),
        )
    
    if config.dataset == "tmqm":
        return tmqm.TMQMDataset(
            root_dir=config.root_dir,
            num_train_molecules=config.num_train_molecules,
            num_val_molecules=config.num_val_molecules,
            num_test_molecules=config.num_test_molecules,
        )

    if config.dataset == "platonic_solids":
        return platonic_solids.PlatonicSolidsDataset(
            train_solids=config.train_solids,
            val_solids=config.val_solids,
            test_solids=config.test_solids,
        )

    if config.dataset == "geom_drugs":
        return geom_drugs.GEOMDrugsDataset(
            root_dir=config.root_dir,
            use_gcdm_splits=config.use_gcdm_splits,
            num_train_molecules=config.num_train_molecules,
            num_val_molecules=config.num_val_molecules,
            num_test_molecules=config.num_test_molecules,
        )

    raise ValueError(
        f"Unknown dataset: {config.dataset}. Available datasets: qm9, platonic_solids, geom_drugs"
    )


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
    logging.info(f"Extracting {path} to {root}")
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
            logging.info(f"Extracting {name} to {root}")
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
