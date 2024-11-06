from typing import List, Iterable, Dict, Set

from absl import logging
import os
import urllib
import numpy as np
import ase
import rdkit.Chem as Chem


from symphony.data import datasets
from symphony import datatypes


QM9_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
)


class QM9Dataset(datasets.InMemoryDataset):
    """QM9 dataset."""

    def __init__(
        self,
        root_dir: str,
        check_molecule_sanity: bool,
        mol_num: int = 0,
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.check_molecule_sanity = check_molecule_sanity
        self.all_structures = None
        self.mol_num = mol_num

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        return np.asarray([1, 6, 7, 8, 9])

    def structures(self) -> Iterable[datatypes.Structures]:
        if self.all_structures is None:
            self.all_structures = load_qm9(self.root_dir, self.check_molecule_sanity)

        return self.all_structures

    def split_indices(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of indices for each split."""
        return {"train": [self.mol_num], "val": [self.mol_num], "test": [self.mol_num]}


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

    path = datasets.utils.download_url(QM9_URL, root_dir)
    datasets.utils.extract_zip(path, root_dir)

    raw_mols_path = os.path.join(root_dir, "gdb9.sdf")
    supplier = Chem.SDMolSupplier(raw_mols_path, removeHs=False, sanitize=False)

    atomic_numbers = QM9Dataset.get_atomic_numbers()
    all_structures = []
    for mol in supplier:
        if mol is None:
            continue

        # Check that the molecule passes some basic checks from Posebusters.
        if check_molecule_sanity:
            sane = molecule_sanity(mol)
            if not sane:
                continue

        # Convert to Structure.
        structure = datatypes.Structures(
            nodes=datatypes.NodesInfo(
                positions=np.asarray(mol.GetConformer().GetPositions()),
                species=np.searchsorted(
                    atomic_numbers,
                    np.asarray([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
                ),
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=None,
            n_node=np.asarray([mol.GetNumAtoms()]),
            n_edge=None,
        )
        all_structures.append(structure)

    return all_structures

