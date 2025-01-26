from typing import List, Iterable, Dict

from absl import logging
import os
import numpy as np
import ase
import tqdm


from symphony.data import datasets
from symphony import datatypes


CATH_URL = "http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/cath-dataset-nonredundant-S20-v4_3_0.pdb.tgz"


class CATHDataset(datasets.InMemoryDataset):
    """CATH dataset."""

    def __init__(
        self,
        root_dir: str,
        num_train_molecules: int,
        num_val_molecules: int,
        num_test_molecules: int,
        train_on_single_molecule: bool = False,
        train_on_single_molecule_index: int = 0,
        rng_seed: int = 6489,  # consistent w foldingdiff
    ):
        super().__init__()

        if root_dir is None:
            raise ValueError("root_dir must be provided.")

        self.root_dir = root_dir
        self.train_on_single_molecule = train_on_single_molecule

        if self.train_on_single_molecule:
            logging.info(
                f"Training on a single molecule with index {train_on_single_molecule_index}."
            )
            self.num_train_molecules = 1
            self.num_val_molecules = 1
            self.num_test_molecules = 1
        else:
            self.num_train_molecules = num_train_molecules
            self.num_val_molecules = num_val_molecules
            self.num_test_molecules = num_test_molecules

        self.all_structures = None
        self.rng = np.random.default_rng(seed=6489)

    @staticmethod
    def get_atomic_numbers() -> np.ndarray:
        # TODO how are we going to keep track of this
        # representing residues by their CB atoms
        return np.asarray([6] * 22 + [6, 6, 7, 7])

    @staticmethod
    def get_amino_acids() -> List[str]:
        return [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "PYL",
            "SEC",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ]

    @staticmethod
    def get_species() -> List[str]:
        return CATHDataset.get_amino_acids() + ["C", "CA", "N", "X"]

    def structures(self) -> Iterable[datatypes.Structures]:
        if self.all_structures is None:
            self.all_structures = load_cath(self.root_dir)

        return self.all_structures

    def split_indices(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of indices for each split."""
        if self.train_on_single_molecule:
            return {
                "train": [self.train_on_single_molecule_index],
                "val": [self.train_on_single_molecule_index],
                "test": [self.train_on_single_molecule_index],
            }

        # using cath splits from foldingdiff
        indices = np.arange(len(self.all_structures))
        self.rng.shuffle(indices)
        splits = {
            "train": np.arange(self.num_train_molecules),
            "val": np.arange(
                self.num_train_molecules,
                self.num_train_molecules + self.num_val_molecules,
            ),
            "test": np.arange(
                self.num_train_molecules + self.num_val_molecules,
                self.num_train_molecules
                + self.num_val_molecules
                + self.num_test_molecules,
            ),
        }
        splits = {k: indices[v] for k, v in splits.items()}
        return splits


def load_cath(
    root_dir: str,
) -> List[ase.Atoms]:
    """Load the CATH dataset."""

    def parse_pdb_format(line):
        # return type, amino acid, coordinates
        return {
            "atom_type": line[13:16].strip(),
            "residue": line[16:20].strip(),
            "x": float(line[31:38].strip()),
            "y": float(line[38:46].strip()),
            "z": float(line[46:55].strip()),
        }

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    path = datasets.utils.download_url(CATH_URL, root_dir)
    datasets.utils.extract_tar(path, root_dir)
    mols_path = os.path.join(root_dir, "dompdb")

    atom_types = ["C", "CA", "N", "X", "CB"]
    amino_acid_abbr = CATHDataset.get_amino_acids()
    all_structures = []

    def _add_structure(pos, spec, molfile):
        assert len(pos) == len(spec), f"Length mismatch: {len(pos)} vs {len(spec)} in {molfile}"
        # foldingdiff does this
        # (also splits anything >128 residues into random 128-residue chunks)
        if len(spec) < 120:
            return

        pos = np.asarray(pos)[:64]
        spec = np.asarray(spec)[:64]

        # Convert to Structure.
        structure = datatypes.Structures(
            nodes=datatypes.NodesInfo(
                positions=pos,
                species=spec,
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=None,
            n_node=np.asarray([len(spec)]),
            n_edge=None,
        )
        all_structures.append(structure)

    logging.info("Loading structures...")
    for mol_file in os.listdir(mols_path):
        mol_path = os.path.join(mols_path, mol_file)
        positions = []
        species = []  # arbitrary ordering: atoms, then amino acids
        last_c_term = None
        first_n = None
        # read pdb
        with open(mol_path, "r") as f:
            for line in f:
                items = parse_pdb_format(line.strip())
                # handle alternate configurations
                if items["residue"][-3:] not in amino_acid_abbr:
                    print(f"Skipping {items['residue']}")
                    positions = []
                    species = []
                    break
                if len(items["residue"]) == 4 and items["residue"][0] == "B":
                    continue  # TODO executive decision on my end to choose btwn alternatives
                # check if this is a different fragment
                if last_c_term is not None and items["atom_type"] == "N":
                    dist_from_last = np.linalg.norm(
                        last_c_term - np.array([items["x"], items["y"], items["z"]])
                    )
                    if dist_from_last > 5.0:  # TODO hardcoded
                        _add_structure(positions, species, mol_file)
                        last_c_term = None
                        first_n = None
                        positions = []
                        species = []
                # take out everything that isn't part of the backbone
                if items["atom_type"] in atom_types:
                    positions.append([items["x"], items["y"], items["z"]])
                    # encode residues as "atoms" located at their beta carbon
                    # GLY just doesn't get anything i guess (TODO ???)
                    if items["atom_type"] == "CB":
                        species.append(amino_acid_abbr.index(items["residue"][-3:]))
                    else:
                        if first_n is None and items["atom_type"] == "N":
                            items["atom_type"] = "X"
                            first_n = np.array([items["x"], items["y"], items["z"]])
                        species.append(22 + atom_types.index(items["atom_type"]))
                        if items["atom_type"] == "C":
                            last_c_term = np.array([items["x"], items["y"], items["z"]])
        # add last structure
        if len(species):
            _add_structure(positions, species, mol_file)
            first_n = None

        
    logging.info(f"Loaded {len(all_structures)} structures.")
    return all_structures
