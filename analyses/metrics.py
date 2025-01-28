"""Metrics for evaluating generative models for molecules."""
from typing import Dict, Tuple, List, Optional, Sequence, Any

import os
import io
import glob
import collections
import functools

from absl import logging
import chex
from collections import Counter
from itertools import groupby
import jax
import jax.numpy as jnp
import numpy as np
import ase
import biotite.structure as struc
from biotite.structure.io import pdb
import tqdm
from rdkit import RDLogger
import rdkit.Chem as Chem
from rdkit.Chem import rdDetermineBonds
import e3nn_jax as e3nn
import posebusters
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tmtools

try:
    # This requires Torch.
    import analyses.edm_analyses.analyze as edm_analyze
except ImportError:
    logging.info("EDM analyses not available.")

try:
    from openbabel import openbabel
    from openbabel import pybel
except ImportError:
    logging.info("OpenBabel not available.")


def xyz_to_rdkit_molecule(molecules_file: str) -> Chem.Mol:
    """Converts a molecule from xyz format to an RDKit molecule."""
    mol = Chem.MolFromXYZFile(molecules_file)
    return Chem.Mol(mol)


def ase_to_rdkit_molecules(ase_mol: Sequence[ase.Atoms]) -> List[Chem.Mol]:
    """Converts molecules from ase format to RDKit molecules."""
    return [ase_to_rdkit_molecule(mol) for mol in ase_mol]


def ase_to_rdkit_molecule(ase_mol: ase.Atoms) -> Chem.Mol:
    """Converts a molecule from ase format to an RDKit molecule."""
    with io.StringIO() as f:
        ase.io.write(f, ase_mol, format="xyz")
        f.seek(0)
        xyz = f.read()
    mol = Chem.MolFromXYZBlock(xyz)
    return Chem.Mol(mol)


def get_all_molecules(molecules_dir: str) -> List[Chem.Mol]:
    """Returns all molecules in a directory."""
    molecules = []
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        mol = xyz_to_rdkit_molecule(molecules_file)
        molecules.append(mol)

    return molecules


def get_all_valid_molecules(molecules: Sequence[Chem.Mol]) -> List[Chem.Mol]:
    """Returns all valid molecules (with bonds inferred)."""
    return [mol for mol in molecules if check_molecule_validity(mol)]


def get_all_valid_molecules_with_openbabel(
    molecules: Sequence[Tuple["openbabel.OBMol", "str"]]
) -> List["openbabel.OBMol"]:
    """Returns all molecules in a directory."""
    return [
        (mol, smiles)
        for mol, smiles in molecules
        if check_molecule_validity_with_openbabel(mol)
    ]


def get_all_molecules_with_openbabel(
    molecules_dir: str,
) -> List[Tuple["openbabel.OBMol", "str"]]:
    """Returns all molecules in a directory."""
    molecules = []
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        for mol in pybel.readfile("xyz", molecules_file):
            molecules.append((mol.OBMol, mol.write("smi").split()[0]))

    return molecules


def compute_molecule_sizes(molecules: Sequence[Chem.Mol]) -> np.ndarray:
    """Computes all of sizes of molecules."""
    return np.asarray([mol.GetNumAtoms() for mol in molecules])


def count_atom_types(
    molecules: Sequence[Chem.Mol], normalize: bool = False
) -> Dict[str, np.ndarray]:
    """Computes the number of atoms of each kind in each valid molecule."""
    atom_counts = collections.defaultdict(lambda: 0)
    for mol in molecules:
        for atom in mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1

    if normalize:
        total = sum(atom_counts.values())
        atom_counts = {atom: count / total for atom, count in atom_counts.items()}

    return dict(atom_counts)


def compute_jensen_shannon_divergence(
    source_dist: Dict[str, float], target_dist: Dict[str, float]
) -> float:
    """Computes the Jensen-Shannon divergence between two distributions."""

    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Computes the KL divergence between two distributions."""
        log_p = np.where(p > 0, np.log(p), 0)
        return (p * log_p - p * np.log(q)).sum()

    # Compute the union of the dictionary keys.
    # We assign a probability of 0 to any key that is not present in a distribution.
    keys = set(source_dist.keys()).union(set(target_dist.keys()))
    source_dist = np.asarray([source_dist.get(key, 0) for key in keys])
    target_dist = np.asarray([target_dist.get(key, 0) for key in keys])

    mean_dist = 0.5 * (source_dist + target_dist)
    return 0.5 * (
        kl_divergence(source_dist, mean_dist) + kl_divergence(target_dist, mean_dist)
    )


@functools.partial(
    jax.jit, static_argnames=("batch_size", "num_batches", "num_kernels")
)
def compute_maximum_mean_discrepancy(
    source_samples: chex.Array,
    target_samples: chex.Array,
    rng: chex.PRNGKey,
    batch_size: int,
    num_batches: int,
    num_kernels: int = 30,
) -> float:
    """
    Calculate the `maximum mean discrepancy distance <https://jmlr.csail.mit.edu/papers/v13/gretton12a.html>` between two lists of samples.
    Adapted from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    """

    def rbf_kernel(X: chex.Array, Y: chex.Array, gamma: float) -> chex.Array:
        """RBF (Gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2))"""
        return jnp.exp(
            -gamma * (jnp.linalg.norm(X[:, None] - Y[None, :], axis=-1) ** 2)
        )

    def mmd_rbf(X: chex.Array, Y: chex.Array, gammas: chex.Array) -> float:
        """MMD using RBF (Gaussian) kernel."""

        def squared_mmd_rbf_kernel(gamma: float) -> float:
            XX = rbf_kernel(X, X, gamma).mean()
            YY = rbf_kernel(Y, Y, gamma).mean()
            XY = rbf_kernel(X, Y, gamma).mean()
            return jnp.abs(XX + YY - 2 * XY)

        return jnp.sqrt(jax.vmap(squared_mmd_rbf_kernel)(gammas).sum())

    def mmd_rbf_batched(
        X: chex.Array, Y: chex.Array, gammas: chex.Array, rng: chex.PRNGKey
    ) -> float:
        """Helper function to compute MMD in batches."""
        X_rng, Y_rng = jax.random.split(rng)
        X_indices = jax.random.randint(
            X_rng, shape=(batch_size,), minval=0, maxval=len(X)
        )
        Y_indices = jax.random.randint(
            Y_rng, shape=(batch_size,), minval=0, maxval=len(Y)
        )
        X_batch, Y_batch = X[X_indices], Y[Y_indices]
        return mmd_rbf(X_batch, Y_batch, gammas)

    X = jnp.asarray(source_samples)
    if len(X.shape) == 1:
        X = X[:, None]

    Y = jnp.asarray(target_samples)
    if len(Y.shape) == 1:
        Y = Y[:, None]

    if batch_size is None:
        batch_size = min(len(X), len(Y))

    # We can only compute the MMD if the number of features is the same.
    assert X.shape[1] == Y.shape[1]

    # We set the kernel widths uniform in logspace.
    gammas = jnp.logspace(-3, 3, num_kernels)

    return jax.vmap(lambda rng: mmd_rbf_batched(X, Y, gammas, rng))(
        jax.random.split(rng, num_batches)
    ).mean()


def check_molecule_validity(mol: Chem.Mol) -> bool:
    """Checks whether a molecule is valid using xyz2mol."""

    # We should only have one conformer.
    assert mol.GetNumConformers() == 1, mol.GetNumConformers()

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except ValueError:
        return False

    if mol.GetNumBonds() == 0:
        return False

    return True


def check_molecule_validity_with_openbabel(
    mol: "openbabel.OBMol",
) -> bool:
    if mol.NumBonds() == 0:
        return False

    # Table of valences for each atom type.
    expected_valences = {
        "H": 1,
        "C": 4,
        "N": 3,
        "O": 2,
        "F": 1,
    }

    invalid = False
    for atom in openbabel.OBMolAtomIter(mol):
        atomic_num = atom.GetAtomicNum()
        atomic_symbol = openbabel.GetSymbol(atomic_num)
        atom_valency = atom.GetExplicitValence()
        if atom_valency != expected_valences[atomic_symbol]:
            invalid = True
            break

    return not invalid


def compute_validity(molecules: Sequence[Chem.Mol]) -> float:
    """Computes the fraction of molecules in a directory that are valid using xyz2mol ."""
    valid_molecules = get_all_valid_molecules(molecules)
    return len(valid_molecules) / len(molecules)


def compute_uniqueness(molecules: Sequence[Chem.Mol]) -> float:
    """Computes the fraction of valid molecules that are unique using SMILES."""
    all_smiles = []
    for mol in get_all_valid_molecules(molecules):
        smiles = Chem.MolToSmiles(mol)
        all_smiles.append(smiles)

    # If there are no valid molecules, return 0.
    if len(all_smiles) == 0:
        return 0.0

    return len(set(all_smiles)) / len(all_smiles)


def compute_uniqueness_with_openbabel(
    molecules: Sequence[Tuple["openbabel.OBMol", "str"]]
) -> float:
    """Computes the fraction of OpenBabel molecules that are unique using SMILES."""
    all_smiles = []
    for _, smiles in get_all_valid_molecules_with_openbabel(molecules):
        all_smiles.append(smiles)

    return len(set(all_smiles)) / len(all_smiles)


def compute_bond_lengths(
    molecules: Sequence[Chem.Mol],
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Collect the lengths for each type of chemical bond in given valid molecular geometries.
    Returns a dictionary where the key is the bond type, and the value is the list of all bond lengths of that bond.
    """
    bond_dists = collections.defaultdict(list)
    for mol in molecules:
        distance_matrix = Chem.Get3DDistanceMatrix(mol)

        if mol.GetNumBonds() == 0:
            raise ValueError("Molecule has no bonds.")

        for bond in mol.GetBonds():
            bond_type = bond.GetBondTypeAsDouble()
            atom_index_1 = bond.GetBeginAtomIdx()
            atom_index_2 = bond.GetEndAtomIdx()
            atom_type_1 = mol.GetAtomWithIdx(atom_index_1).GetSymbol()
            atom_type_2 = mol.GetAtomWithIdx(atom_index_2).GetSymbol()
            atom_type_1, atom_type_2 = min(atom_type_1, atom_type_2), max(
                atom_type_1, atom_type_2
            )
            bond_length = distance_matrix[atom_index_1, atom_index_2]
            bond_dists[(atom_type_1, atom_type_2, bond_type)].append(bond_length)

    return {
        bond_type: np.asarray(bond_lengths)
        for bond_type, bond_lengths in bond_dists.items()
    }


def compute_local_environments(
    molecules: Sequence[Chem.Mol], max_num_molecules: int
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Collect the number of distinct local environments given valid molecular geometries.
    Returns a dictionary where the key is the central atom, and the value is a dictionary of counts of distinct local environments.
    """
    local_environments = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )

    for mol_counter, mol in enumerate(molecules):
        if mol_counter == max_num_molecules:
            break

        counts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        for bond in mol.GetBonds():
            atom_index_1 = bond.GetBeginAtomIdx()
            atom_index_2 = bond.GetEndAtomIdx()
            atom_type_1 = mol.GetAtomWithIdx(atom_index_1).GetSymbol()
            atom_type_2 = mol.GetAtomWithIdx(atom_index_2).GetSymbol()
            counts[atom_index_1][atom_type_2] += 1
            counts[atom_index_2][atom_type_1] += 1

        for atom_index, neighbors in counts.items():
            central_atom_type = mol.GetAtomWithIdx(atom_index).GetSymbol()
            neighbors_as_string = ",".join(
                [f"{neighbor}{count}" for neighbor, count in sorted(neighbors.items())]
            )
            local_environments[central_atom_type][neighbors_as_string] += 1

        mol_counter += 1

    return {
        central_atom_type: dict(
            sorted(neighbors.items(), reverse=True, key=lambda x: x[1])
        )
        for central_atom_type, neighbors in local_environments.items()
    }


@functools.partial(jax.jit, static_argnames=("lmax",))
def bispectrum(neighbor_positions: jnp.ndarray, lmax: int) -> float:
    """Computes the bispectrum of a set of neighboring positions."""
    assert neighbor_positions.shape == (neighbor_positions.shape[0], 3)
    x = e3nn.sum(
        e3nn.s2_dirac(neighbor_positions, lmax=lmax, p_val=1, p_arg=-1), axis=0
    )
    rtp = e3nn.reduced_symmetric_tensor_product_basis(
        x.irreps, 3, keep_ir=["0e", "0o"], _use_optimized_implementation=True
    )
    return jnp.einsum("ijkz,i,j,k->z", rtp.array, x.array, x.array, x.array)


def compute_bispectra_of_local_environments(
    molecules: Sequence[Chem.Mol], lmax: int, max_num_molecules: int
) -> Dict[Tuple[str, str], jnp.ndarray]:
    """
    Computes the bispectrum of the local environments given valid molecular geometries.
    Returns a dictionary where the key is the central atom, and the value is a dictionary of bispectra of distinct local environments.
    """
    bispectra = collections.defaultdict(list)
    relative_positions = collections.defaultdict(list)

    for mol_counter, mol in tqdm.tqdm(enumerate(molecules)):
        if mol_counter == max_num_molecules:
            break

        counts = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
        neighbor_positions = collections.defaultdict(list)
        for bond in mol.GetBonds():
            atom_index_1 = bond.GetBeginAtomIdx()
            atom_index_2 = bond.GetEndAtomIdx()
            atom_type_1 = mol.GetAtomWithIdx(atom_index_1).GetSymbol()
            atom_type_2 = mol.GetAtomWithIdx(atom_index_2).GetSymbol()

            counts[atom_index_1][atom_type_2] += 1
            counts[atom_index_2][atom_type_1] += 1
            neighbor_positions[atom_index_1].append(
                mol.GetConformer().GetAtomPosition(atom_index_2)
                - mol.GetConformer().GetAtomPosition(atom_index_1)
            )
            neighbor_positions[atom_index_2].append(
                mol.GetConformer().GetAtomPosition(atom_index_1)
                - mol.GetConformer().GetAtomPosition(atom_index_2)
            )

        neighbor_positions = {
            atom_index: jnp.asarray(positions)
            for atom_index, positions in neighbor_positions.items()
        }
        neighbor_bispectra = {
            atom_index: bispectrum(positions, lmax)
            for atom_index, positions in neighbor_positions.items()
        }

        for atom_index, neighbors in counts.items():
            central_atom_type = mol.GetAtomWithIdx(atom_index).GetSymbol()
            neighbors_as_string = ",".join(
                [f"{neighbor}{count}" for neighbor, count in sorted(neighbors.items())]
            )
            bispectra[(central_atom_type, neighbors_as_string)].append(
                neighbor_bispectra[atom_index]
            )
            relative_positions[(central_atom_type, neighbors_as_string)].append(
                neighbor_positions[atom_index]
            )

    return {
        environment: jnp.asarray(bispectra)
        for environment, bispectra in bispectra.items()
    }, {
        environment: jnp.asarray(relative_positions)
        for environment, relative_positions in relative_positions.items()
    }


def compute_maximum_mean_discrepancies(
    source_samples_dict: Dict[Any, jnp.ndarray],
    target_samples_dict: Dict[Any, jnp.ndarray],
    rng: chex.PRNGKey,
    batch_size: int,
    num_batches: int,
) -> Dict[Any, float]:
    """
    Compute the maximum mean discrepancy distance for each key in the source and target dictionaries.
    """
    results = {}
    for key in source_samples_dict:
        if key not in target_samples_dict:
            continue

        mmd_rng, rng = jax.random.split(rng)
        results[key] = compute_maximum_mean_discrepancy(
            source_samples_dict[key],
            target_samples_dict[key],
            mmd_rng,
            batch_size,
            num_batches,
        )

    return results


def compute_bond_lengths(
    molecules: Sequence[Chem.Mol],
) -> Dict[Tuple[str, str, float], np.ndarray]:
    """
    Collect the lengths for each type of chemical bond in given molecular geometries.
    Returns a dictionary where the key is the bond type, and the value is the list of all bond lengths of that bond.
    """
    bond_dists = collections.defaultdict(list)
    for mol in molecules:
        distance_matrix = Chem.Get3DDistanceMatrix(mol)
        if mol.GetNumBonds() == 0:
            print(mol, mol.GetNumBonds(), mol.GetNumAtoms())
            raise ValueError("Molecule has no bonds.")
        for bond in mol.GetBonds():
            bond_type = bond.GetBondTypeAsDouble()
            atom_index_1 = bond.GetBeginAtomIdx()
            atom_index_2 = bond.GetEndAtomIdx()
            atom_type_1 = mol.GetAtomWithIdx(atom_index_1).GetSymbol()
            atom_type_2 = mol.GetAtomWithIdx(atom_index_2).GetSymbol()
            atom_type_1, atom_type_2 = min(atom_type_1, atom_type_2), max(
                atom_type_1, atom_type_2
            )
            bond_length = distance_matrix[atom_index_1, atom_index_2]
            bond_dists[(atom_type_1, atom_type_2, bond_type)].append(bond_length)

    return {
        bond_type: np.asarray(bond_lengths)
        for bond_type, bond_lengths in bond_dists.items()
    }


def get_posebusters_results(
    molecules: Sequence[Chem.Mol], full_report: bool = False
) -> pd.DataFrame:
    """Returns the results from Posebusters (https://github.com/maabuu/posebusters)."""
    return posebusters.PoseBusters(config="mol").bust(
        mol_pred=molecules, full_report=full_report
    )


def check_backbone_validity(mol: struc.AtomArray) -> bool:
    '''ok the silliest easiest check is to look for -N-C-C- repeat'''
    atoms = mol.atom_name
    n_count = np.array([a == "N" for a in atoms]).sum()
    c_count = np.array([a == "C" for a in atoms]).sum()
    ca_count = np.array([a == "CA" for a in atoms]).sum()
    return n_count == c_count and c_count == ca_count


def compute_backbone_validity(structure_list: Sequence[struc.AtomArrayStack]) -> float:
    '''Computes the percentage of given structures that have a valid protein backbone'''
    return sum([check_backbone_validity(s.get_array(0)) for s in structure_list]) / len(structure_list)


def compute_backbone_uniqueness(structure_list: Sequence[struc.AtomArrayStack]) -> float:
    '''Computes the percentage of unique structures in the given list, 
    where uniqueness is defined by residue sequence'''
    sequences = []
    for s in structure_list:
        try:
            sequences.append(str(struc.to_sequence(s)[0][0]))
        except:
            sequences.append("")
    return len(set(sequences)) / len(sequences)


def get_ramachandran_plot(
        structure: struc.AtomArray,
        window_size: Tuple[int, int] = (300, 300),
    ):
    phi_angles, psi_angles, _ = struc.dihedral_backbone(structure.get_array(0))
    phi_angles = phi_angles[1:-1][:300]
    psi_angles = psi_angles[1:-1][:300]
    phi_angles = np.rad2deg(phi_angles)
    psi_angles = np.rad2deg(psi_angles)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=phi_angles, y=psi_angles, mode="markers", marker=dict(size=5, color="blue")))
    fig.add_trace(go.Scatter(x=[-180, 180], y=[0, 0], mode="lines", line=dict(color="black", width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=[0, 0], y=[-180, 180], mode="lines", line=dict(color="black", width=1, dash="dash")))

    fig.update_layout(
        xaxis_title=r"${\phi}$",
        yaxis_title=r"${\psi}$",
        xaxis_range=[-180, 180],
        yaxis_range=[-180, 180],
        xaxis_tick0=-180,
        xaxis_dtick=45,
        yaxis_tick0=-180,
        yaxis_dtick=45,
        xaxis_tickfont=dict(size=12),
        yaxis_tickfont=dict(size=12),
        width=window_size[0],
        height=window_size[1],
    )

    return fig


def get_ramachandran_plots(
    structure_list: Sequence[struc.AtomArrayStack],
    window_size: Tuple[int, int] = (300, 300),
    rows: int = 2,
    cols: int = 4,
):
    bigfig = make_subplots(rows=rows, cols=cols)
    for i, structure in enumerate(structure_list):
        fig = get_ramachandran_plot(structure)
        row = i // cols + 1
        col = i % cols + 1
        bigfig.add_trace(fig.data[0], row=row, col=col)
        bigfig.update_xaxes(fig.layout.xaxis, row=row, col=col)
        bigfig.update_yaxes(fig.layout.yaxis, row=row, col=col)
    bigfig.update_layout(
        margin=dict(l=80, r=80, t=80, b=80),
        width=window_size[0]*cols+80*2,
        height=window_size[1]+80*2,
        showlegend=False,
    )
    return bigfig


def count_secondary_structures(structure: struc.AtomArray) -> Tuple[int, int]:
    """Count the secondary structures (# alpha, # beta) in the given pdb file.
    Adapted from foldingdiff: https://github.com/microsoft/foldingdiff/tree/main"""
    # a = alpha helix, b = beta sheet, c = coil
    ss = struc.annotate_sse(structure)
    # https://stackoverflow.com/questions/6352425/whats-the-most-pythonic-way-to-identify-consecutive-duplicates-in-a-list
    ss_grouped = ss_grouped = {k: sum(1 for _ in g) for k, g in groupby(sorted(ss))}

    num_alpha = ss_grouped.get("a", 0)
    num_beta = ss_grouped.get("b", 0)

    num_residues = struc.get_residue_count(structure)

    return num_alpha/num_residues, num_beta/num_residues


def count_secondary_structures_multi(structures: List[struc.AtomArrayStack]) -> Tuple[List[int], List[int]]:
    """Count the secondary structures (# alpha, # beta) in the given protein structures."""
    num_alpha = []
    num_beta = []
    for structure in structures:
        a, b = count_secondary_structures(structure.get_array(0))
        num_alpha.append(a)
        num_beta.append(b)
    return num_alpha, num_beta


def compute_tm_score(struct1: struc.AtomArray, struct2: struc.AtomArray) -> float:
    """Compute the TM score between two structures."""
    coords1 = struct1.get_coord()
    coords2 = struct2.get_coord()
    seq1 = struc.to_sequence(struct1)[0][0]
    seq2 = struc.to_sequence(struct2)[0][0]
    return tmtools.tm_score(coords1, coords2, str(seq1), str(seq2))


def get_all_edm_analyses_results(
    molecules_basedir: str, extract_hyperparams_from_path: bool, read_as_sdf: bool
) -> pd.DataFrame:
    """Returns the EDM analyses results for the given directories as a pandas dataframe, keyed by path."""

    def find_in_path_fn(string):
        """Returns a function that finds a substring in a path."""

        def find_in_path(path):
            occurrences = [
                subs[len(string) :]
                for subs in path.split("/")
                if subs.startswith(string)
            ]
            if len(occurrences):
                return occurrences[0]

        return find_in_path

    def find_model_in_path(path, all_models: Optional[Sequence[str]] = None):
        """Returns the model name from the path."""
        if all_models is None:
            all_models = ["nequip", "e3schnet", "mace", "marionette"]

        for subs in path.split("/"):
            for model in all_models:
                if model in subs:
                    return model

    # Find all directories containing molecules.
    molecules_dirs = set()
    suffix = "*.sdf" if read_as_sdf else "*.xyz"
    for molecules_file in glob.glob(
        os.path.join(molecules_basedir, "**", suffix), recursive=True
    ):
        molecules_dirs.add(os.path.dirname(molecules_file))

    if not len(molecules_dirs):
        raise ValueError(f"No molecules found in {molecules_basedir}.")

    # Analyze each directory.
    results = pd.DataFrame()
    for molecules_dir in molecules_dirs:
        metrics_df = get_edm_analyses_results(molecules_dir, read_as_sdf)
        results = pd.concat([results, metrics_df], ignore_index=True)

    # Extract hyperparameters from path.
    if extract_hyperparams_from_path:
        paths = results["path"]
        for hyperparam, substring, dtype in [
            ("config.num_interactions", "interactions=", int),
            ("max_l", "l=", int),
            (
                "config.target_position_predictor.num_channels",
                "position_channels=",
                int,
            ),
            ("config.num_channels", "channels=", int),
            ("focus_and_atom_type_inverse_temperature", "fait=", float),
            ("position_inverse_temperature", "pit=", float),
            ("step", "step=", str),
            ("global_embedding", "global_embed=", str),
        ]:
            results[hyperparam] = paths.apply(find_in_path_fn(substring)).astype(dtype)
        results["model"] = paths.apply(find_model_in_path)

    return results


def get_edm_analyses_results(molecules_dir: str, read_as_sdf: bool) -> pd.DataFrame:
    """Returns the EDM analyses results for the given directory as a pandas dataframe."""
    # Disable RDKit logging.
    RDLogger.DisableLog("rdApp.info")
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)

    metrics = edm_analyze.analyze_stability_for_molecules_in_dir(
        molecules_dir, read_as_sdf=read_as_sdf
    )
    return pd.DataFrame().from_dict(
        {"path": molecules_dir, **{key: [val] for key, val in metrics.items()}}
    )
