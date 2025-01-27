from typing import List, Iterable, Dict

from absl import logging
import os
import numpy as np
import ase
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import warnings

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
    def atoms_to_species() -> Dict[str, int]:
        mapping = {}
        mapping["C"] = 0
        mapping["CA"] = 1
        amino_acid_abbr = CATHDataset.get_amino_acids()
        for i, aa in enumerate(amino_acid_abbr):
            mapping[aa] = i + 2
        mapping["N"] = 24
        mapping["X"] = 25
        return mapping


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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
                min(
                    len(self.all_structures),
                    (self.num_train_molecules
                        + self.num_val_molecules
                        + self.num_test_molecules),
                )
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

    all_structures = []

    def _add_structure(pos, spec, molfile, residue_starts):
        assert len(pos) == len(spec), f"Length mismatch: {len(pos)} vs {len(spec)} in {molfile}"
        # foldingdiff does this
        # (also splits anything >128 residues into random 128-residue chunks)
        if len(spec) < 120:
            return

        pos = np.asarray(pos)
        spec = np.asarray(spec)

        # Convert to Structure.
        structure = datatypes.Structures(
            nodes=datatypes.NodesInfo(
                positions=pos,
                species=spec,
            ),
            edges=None,
            receivers=None,
            senders=None,
            globals=datatypes.GlobalsInfo(
                num_residues=np.asarray([len(residue_starts)]),
                residue_starts=residue_starts,
            ),
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
        f = pdb.PDBFile.read(mol_path)
        structure = pdb.get_structure(f)
        backbone = structure.get_array(0)
        mask = np.isin(backbone.atom_name, ["CA", "N", "C", "CB"])
        backbone = backbone[mask]
        fragment_starts = np.concatenate([
            np.array([0]),
            # distance between CB and N is ~2.4 angstroms + some wiggle room
            struc.check_backbone_continuity(backbone, max_len=2.6),
            np.array([len(backbone)]),
        ])
        for i in range(len(fragment_starts) - 1):
            fragment = backbone[fragment_starts[i]:fragment_starts[i + 1]]
            try:
                first_n = np.argwhere(fragment.atom_name == "N")[0][0]
                positions = fragment.coord
                species = fragment.atom_name
                species[first_n] = "X"
                # set CB to corresponding residue name
                cb_atoms = np.argwhere(fragment.atom_name == "CB").flatten()
                species[cb_atoms] = fragment.res_name[cb_atoms]
                species = np.vectorize(CATHDataset.atoms_to_species().get)(species)
                residue_starts = struc.get_residue_starts(fragment)
                _add_structure(positions, species, mol_file, residue_starts)
            except:
                continue

        
    logging.info(f"Loaded {len(all_structures)} structures.")
    return all_structures
