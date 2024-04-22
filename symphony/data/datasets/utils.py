import ase
import matscipy.neighbours
import numpy as np
import ml_collections

from symphony.data.datasets import dataset, platonic_solids, qm9, tmqm
from symphony import datatypes


def get_dataset(config: ml_collections.ConfigDict) -> dataset.InMemoryDataset:
    """Creates the dataset of structures, as specified in the config."""

    if config.dataset == "qm9":
        return qm9.QM9Dataset(
            root_dir=config.root_dir,
            check_molecule_sanity=config.get("check_molecule_sanity", False),
            use_edm_splits=config.get("use_edm_splits", False),
            num_train_molecules=config.get("num_train_molecules"),
            num_val_molecules=config.get("num_val_molecules"),
            num_test_molecules=config.get("num_test_molecules"),
        )
    
    if config.dataset == "platonic_solids":
        return platonic_solids.PlatonicSolidsDataset(
            train_solids=config.train_solids,
            val_solids=config.val_solids,
            test_solids=config.test_solids,
        )
    
    if config.dataset == "tmqm":
        return tmqm.TMQMDataset(
            root_dir=config.root_dir,
            num_train_molecules=config.get("num_train_molecules"),
            num_val_molecules=config.get("num_val_molecules"),
            num_test_molecules=config.get("num_test_molecules"),
        )

    raise ValueError(f"Unknown dataset: {config.dataset}")
