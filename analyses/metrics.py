"""Metrics for evaluating generative models for molecules."""
from typing import Dict, Tuple, List

import chex
import collections
import functools
import jax
import jax.numpy as jnp
import numpy as np
import os
import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import rdDetermineBonds
import e3nn_jax as e3nn
import posebusters
import pandas as pd
from absl import logging

try:
    # This requires Torch.
    import analyses.edm_analyses.analyze as edm_analyze
except ImportError:
    logging.info("EDM analyses not available.")


def xyz_to_rdkit_molecule(molecules_file: str) -> Chem.Mol:
    """Converts a molecule from xyz format to an RDKit molecule."""
    mol = Chem.MolFromXYZFile(molecules_file)
    return Chem.Mol(mol)


def count_molecule_sizes(molecules_dir: str) -> np.ndarray:
    """Computes the distribution of sizes for valid molecules."""
    sizes = []
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        if not check_molecule_validity(molecules_file):
            continue

        mol = xyz_to_rdkit_molecule(molecules_file)
        sizes.append(mol.GetNumAtoms())

    return np.asarray(sizes)


def count_atom_types(molecules_dir: str, normalize: bool = False) -> Dict[str, np.ndarray]:
    """Computes the number of atoms of each kind in each valid molecule."""
    atom_counts = collections.defaultdict(lambda: 0)
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        if not check_molecule_validity(molecules_file):
            continue

        mol = xyz_to_rdkit_molecule(molecules_file)
        for atom in mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1

    if normalize:
        total = sum(atom_counts.values())
        atom_counts = {atom: count / total for atom, count in atom_counts.items()}
    
    return dict(atom_counts)


@functools.partial(jax.jit, static_argnames=("batch_size", "num_batches"))
def compute_maximum_mean_discrepancy(
    source_samples: chex.Array,
    target_samples: chex.Array,
    rng: chex.PRNGKey,
    batch_size: int,
    num_batches: int,
) -> float:
    """
    Calculate the `maximum mean discrepancy distance <https://jmlr.csail.mit.edu/papers/v13/gretton12a.html>`_ between two lists of samples.
    Adapted from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    """

    def rbf_kernel(X: chex.Array, Y: chex.Array, gamma: float) -> chex.Array:
        """RBF (Gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2))"""
        return jnp.exp(
            -gamma * (jnp.linalg.norm(X[:, None] - Y[None, :], axis=-1) ** 2)
        )

    def mmd_rbf(X: chex.Array, Y: chex.Array, gammas: chex.Array) -> float:
        """MMD using RBF (Gaussian) kernel."""
        XX = jax.vmap(lambda gamma: rbf_kernel(X, X, gamma))(gammas).sum(axis=0)
        YY = jax.vmap(lambda gamma: rbf_kernel(Y, Y, gamma))(gammas).sum(axis=0)
        XY = jax.vmap(lambda gamma: rbf_kernel(X, Y, gamma))(gammas).sum(axis=0)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def mmd_rbf_batched(
        X: chex.Array, Y: chex.Array, gammas: chex.Array, rng: chex.PRNGKey
    ) -> float:
        """Helper function to compute MMD in batches."""
        X_rng, Y_rng = jax.random.split(rng)
        X_indices = jax.random.randint(
            X_rng, shape=(batch_size,), minval=0, maxval=len(X)
        )
        X_batch = X[X_indices]
        Y_indices = jax.random.randint(
            Y_rng, shape=(batch_size,), minval=0, maxval=len(Y)
        )
        Y_batch = Y[Y_indices]
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

    gammas = jnp.logspace(-2, 2, 30)
    mmd_estimates = jax.vmap(lambda rng: mmd_rbf_batched(X, Y, gammas, rng))(
        jax.random.split(rng, num_batches)
    )
    return mmd_estimates.mean()


def check_molecule_validity(molecules_file: str) -> bool:
    """Checks whether a molecule is valid using xyz2mol."""
    mol = Chem.MolFromXYZFile(molecules_file)
    mol = Chem.Mol(mol)

    # We should only have one conformer.
    assert mol.GetNumConformers() == 1

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except ValueError:
        return False

    return True


def compute_validity(molecules_dir: str) -> float:
    """Computes the fraction of molecules in a directory that are valid using xyz2mol ."""
    valid = total = 0
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        total += 1
        molecules_file = os.path.join(molecules_dir, molecules_file)
        valid += check_molecule_validity(molecules_file)

    return valid / total


def compute_bond_lengths(molecules_dir: str) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Collect the lengths for each type of chemical bond in given valid molecular geometries.
    Returns a dictionary where the key is the bond type, and the value is the list of all bond lengths of that bond.
    """
    bond_dists = collections.defaultdict(list)
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        if not check_molecule_validity(molecules_file):
            continue

        mol = xyz_to_rdkit_molecule(molecules_file)

        # This will work as the molecule is valid.
        rdDetermineBonds.DetermineBonds(mol, charge=0)

        distance_matrix = Chem.Get3DDistanceMatrix(mol)
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
    molecules_dir: str, max_num_molecules: int
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Collect the number of distinct local environments given valid molecular geometries.
    Returns a dictionary where the key is the central atom, and the value is a dictionary of counts of distinct local environments.
    """
    local_environments = collections.defaultdict(
        lambda: collections.defaultdict(lambda: 0)
    )

    count = 0
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        if not check_molecule_validity(molecules_file):
            continue

        mol = xyz_to_rdkit_molecule(molecules_file)

        # This will work as the molecule is valid.
        rdDetermineBonds.DetermineBonds(mol, charge=0)

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

        count += 1
        if count == max_num_molecules:
            break

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
    molecules_dir: str, lmax: int, max_num_molecules: int
) -> Dict[Tuple[str, str], jnp.ndarray]:
    """
    Computes the bispectrum of the local environments given valid molecular geometries.
    Returns a dictionary where the key is the central atom, and the value is a dictionary of bispectra of distinct local environments.
    """
    bispectra = collections.defaultdict(list)

    count = 0
    for molecules_file in tqdm.tqdm(os.listdir(molecules_dir)):
        if not molecules_file.endswith(".xyz"):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        if not check_molecule_validity(molecules_file):
            continue

        mol = xyz_to_rdkit_molecule(molecules_file)

        # This will work as the molecule is valid.
        rdDetermineBonds.DetermineBonds(mol, charge=0)

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

        count += 1
        if count == max_num_molecules:
            break

    return {
        environment: jnp.asarray(bispectra)
        for environment, bispectra in bispectra.items()
    }


def compute_maximum_mean_discrepancies_for_bispectra(
    source_bispectra: Dict[Tuple[str, str], jnp.ndarray],
    target_bispectra: Dict[Tuple[str, str], jnp.ndarray],
    rng: chex.PRNGKey,
    batch_size: int,
    num_batches: int,
) -> Dict[Tuple[str, str], float]:
    """
    Compute the maximum mean discrepancy distance between the bispectra distributions of two sets of molecules.
    """
    results = {}
    for environment in source_bispectra:
        if environment not in target_bispectra:
            continue

        mmd_rng, rng = jax.random.split(rng)
        mmd = compute_maximum_mean_discrepancy(
            source_bispectra[environment],
            target_bispectra[environment],
            mmd_rng,
            batch_size,
            num_batches,
        )

        central_atom_type, neighbors = environment
        print(
            f"The MMD distance of bispectra for central atom {central_atom_type} and neighbors {neighbors} is {mmd:0.5f}"
        )

        results[environment] = mmd

    return results


def compute_bond_lengths(
    molecules_dir: str,
) -> Dict[Tuple[str, str, float], np.ndarray]:
    """
    Collect the lengths for each type of chemical bond in given valid molecular geometries.
    Returns a dictionary where the key is the bond type, and the value is the list of all bond lengths of that bond.
    """
    bond_dists = collections.defaultdict(list)
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        molecules_file = os.path.join(molecules_dir, molecules_file)
        if not check_molecule_validity(molecules_file):
            continue

        mol = xyz_to_rdkit_molecule(molecules_file)

        # This will work as the molecule is valid.
        rdDetermineBonds.DetermineBonds(mol, charge=0)

        distance_matrix = Chem.Get3DDistanceMatrix(mol)
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


def compute_maximum_mean_discrepancies_for_bond_lengths(
    source_bond_dists: Dict[Tuple[str, str, float], np.ndarray],
    target_bond_dists: Dict[Tuple[str, str, float], np.ndarray],
    rng: chex.PRNGKey,
    batch_size: int,
    num_batches: int,
) -> Dict[Tuple[str, str, float], float]:
    """
    Compute the maximum mean discrepancy distance between the bond length distributions of two sets of molecules.
    """
    results = {}
    for bond_key in sorted(source_bond_dists):
        if bond_key not in target_bond_dists:
            continue

        mmd_rng, rng = jax.random.split(rng)
        mmd = compute_maximum_mean_discrepancy(
            source_bond_dists[bond_key],
            target_bond_dists[bond_key],
            mmd_rng,
            batch_size,
            num_batches,
        )

        atom_type_1, atom_type_2, bond_type = bond_key
        print(
            f"The MMD distance of {atom_type_1}-{atom_type_2} (bond type {bond_type}) bond length distributions is {mmd:0.5f}"
        )

        results[bond_key] = mmd

    return results


def _get_sdf_files(molecules_dir: str) -> List[str]:
    """Returns all .sdf molecule files in a directory."""
    files = []
    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".sdf"):
            continue
        molecules_file = os.path.join(molecules_dir, molecules_file)
        files.append(molecules_file)
    return files


def get_posebusters_results(molecules_sdf_dir: str, full_report: bool = False) -> pd.DataFrame:
    """Returns the results from Posebusters (https://github.com/maabuu/posebusters)."""
    return posebusters.PoseBusters(config="mol").bust(mol_pred=_get_sdf_files(molecules_sdf_dir), full_report=full_report)


def get_edm_analyses_results(
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
        metrics = edm_analyze.analyze_stability_for_molecules_in_dir(
            molecules_dir, read_as_sdf=read_as_sdf
        )
        metrics_df = pd.DataFrame().from_dict(
            {"path": molecules_dir, **{key: [val] for key, val in metrics.items()}}
        )
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
