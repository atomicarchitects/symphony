##########################################################
# From G-SchNet repository                               #
# https://github.com/atomistic-machine-learning/G-SchNet #
##########################################################

import numpy as np
from openbabel import openbabel as ob
from openbabel import pybel

from analyses import analysis


def _create_mol_dict(mols, idcs=None):
    """
    Create a dictionary holding indices of a list of molecules where the key is a
    string that represents the atomic composition (i.e. the number of atoms per type in
    the molecule, e.g. H2C3O1, where the order of types is determined by increasing
    nuclear charge). This is especially useful to speed up the comparison of molecules
    as candidate structures with the same composition of atoms can easily be accessed
    while ignoring all molecules with different compositions.

    Args:
        mols (list of utility_classes.Molecule or numpy.ndarray): the molecules or
            the atomic numbers of the molecules which are referenced in the dictionary
        idcs (list of int, optional): indices of a subset of the molecules in mols that
            shall be put into the dictionary (if None, all structures in mol will be
            referenced in the dictionary, default: None)

    Returns:
        dict (str->list of int): dictionary with the indices of molecules in mols
            ordered by their atomic composition
    """
    if idcs is None:
        idcs = range(len(mols))
    mol_dict = {}
    for idx in idcs:
        mol = mols[idx]
        mol_key = _get_atoms_per_type_str(mol)
        mol_dict = _update_dict(mol_dict, key=mol_key, val=idx)
    return mol_dict


def _update_dict(old_dict, **kwargs):
    """
    Update an existing dictionary (any->list of any) with new entries where the new
    values are either appended to the existing lists if the corresponding key already
    exists in the dictionary or a new list under the new key is created.

    Args:
        old_dict (dict (any->list of any)): original dictionary that shall be updated
        **kwargs: keyword arguments that can either be a dictionary of the same format
            as old_dict (new_dict=dict (any->list of any)) which will be merged into
            old_dict or a single key-value pair that shall be added (key=any, val=any)

    Returns:
        dict (any->list of any): the updated dictionary
    """
    if "new_dict" in kwargs:
        for key in kwargs["new_dict"]:
            if key in old_dict:
                old_dict[key] += kwargs["new_dict"][key]
            else:
                old_dict[key] = kwargs["new_dict"][key]
    if "val" in kwargs and "key" in kwargs:
        if kwargs["key"] in old_dict:
            old_dict[kwargs["key"]] += [kwargs["val"]]
        else:
            old_dict[kwargs["key"]] = [kwargs["val"]]
    return old_dict


def _get_atoms_per_type_str(mol, type_infos={1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}):
    """
    Get a string representing the atomic composition of a molecule (i.e. the number
    of atoms per type in the molecule, e.g. H2C3O1, where the order of types is
    determined by increasing nuclear charge).

    Args:
        mol (ase.Atoms): molecule

    Returns:
        str: the atomic composition of the molecule
    """
    n_atoms_per_type = np.bincount(mol.numbers, minlength=10)
    s = ""
    for t, n in zip(type_infos.keys(), n_atoms_per_type):
        s += f"{type_infos[t]}{int(n):d}"
    return s


def get_fingerprint(ase_mol, use_bits=False):
    """
    Compute the molecular fingerprint (Open Babel FP2), canonical smiles
    representation, and number of atoms per type (e.g. H2O1) of a molecule.

    Args:
        ase_mol (ase.Atoms): molecule
        use_bits (bool, optional): set True to return the non-zero bits in the
            fingerprint instead of the pybel.Fingerprint object (default: False)

    Returns:
        pybel.Fingerprint or set of int: the fingerprint of the molecule or a set
            containing the non-zero bits of the fingerprint if use_bits=True
        str: the atom types contained in the molecule followed by number of
            atoms per type, e.g. H2C3O1, ordered by increasing atom type (nuclear
            charge)
    """
    mol = analysis.construct_pybel_mol(ase_mol)
    # use pybel to get fingerprint
    if use_bits:
        return (
            {*mol.calcfp().bits},
            _get_atoms_per_type_str(ase_mol),
        )
    else:
        return mol.calcfp(), _get_atoms_per_type_str(ase_mol)


def fingerprints_similar(mol1, fp2, symbols2, use_bits=False):
    fp1, symbols1 = get_fingerprint(mol1, use_bits=False)
    if tanimoto_similarity(fp1, fp2, use_bits=use_bits) >= 1:
        # compare canonical smiles representation
        mirror1 = get_mirror_can(mol1)
        return symbols1 == symbols2 or mirror1 == symbols2
    return False


def tanimoto_similarity(mol, other_mol, use_bits=True):
    """
    Get the Tanimoto (fingerprint) similarity to another molecule.

    Args:
     mol (pybel.Fingerprint/list of bits set):
        representation of the second molecule
     other_mol (pybel.Fingerprint/list of bits set):
        representation of the second molecule
     use_bits (bool, optional): set True to calculate Tanimoto similarity
        from bits set in the fingerprint (default: True)

    Returns:
         float: Tanimoto similarity to the other molecule
    """
    if use_bits:
        n_equal = len(mol.intersection(other_mol))
        if len(mol) + len(other_mol) == 0:  # edge case with no set bits
            return 1.0
        return n_equal / (len(mol) + len(other_mol) - n_equal)
    else:
        return mol | other_mol


def get_mirror_can(mol):
    """
    Retrieve the canonical SMILES representation of the mirrored molecule (the
    z-coordinates are flipped).

    Args:
        mol (ase.Atoms): molecule

    Returns:
         String: canonical SMILES string of the mirrored molecule
    """
    # flip z to mirror molecule using x-y plane
    flipped = analysis.construct_obmol(mol)
    for atom in ob.OBMolAtomIter(flipped):
        x, y, z = atom.x(), atom.y(), atom.z()
        atom.SetVector(x, y, -z)
    flipped.ConnectTheDots()
    flipped.PerceiveBondOrders()

    # calculate canonical SMILES of mirrored molecule
    mirror_can = pybel.Molecule(flipped).write("can")
    return mirror_can
