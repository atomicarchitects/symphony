"""Relaxes structures using Universal Force Fields from RDKit."""

from typing import Optional, Tuple, Sequence

import os
import glob

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

import sys

sys.path.append("..")

import analyses.analysis as analysis


def add_bonds_to_molecules(molecules_dir: str, output_dir: str) -> None:
    """Adds bonds to all molecules in a directory."""

    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".xyz"):
            continue

        mol = Chem.MolFromXYZFile(os.path.join(molecules_dir, molecules_file))
        mol = Chem.Mol(mol)
        valid_charge = False
        for charge in [0, -1, 1, -2, 2]:
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=charge)
                valid_charge = True
                break
            except ValueError:
                continue

        if not valid_charge:
            logging.info("Could not find valid charge for %s", molecules_file)
            continue

        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
            elif bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)

        # Save the bonded molecules.
        output_molecules_file = molecules_file.replace(".xyz", ".sdf")
        writer = Chem.SDWriter(os.path.join(output_dir, output_molecules_file))
        writer.write(mol, confId=0)


def relax_structures_of_molecules(molecules_dir: str, output_dir: str) -> None:
    """Relaxes the structures of all molecules in a directory."""

    for molecules_file in os.listdir(molecules_dir):
        if not molecules_file.endswith(".sdf"):
            continue

        mol = Chem.SDMolSupplier(
            os.path.join(molecules_dir, molecules_file), removeHs=False
        )[0]
        energy_value_initial = AllChem.UFFGetMoleculeForceField(mol).CalcEnergy()

        AllChem.UFFOptimizeMolecule(
            mol, maxIters=100000, ignoreInterfragInteractions=False
        )

        energy_value_final = AllChem.UFFGetMoleculeForceField(mol).CalcEnergy()
        logging.info(
            "Energy change for %s: %f",
            molecules_file,
            energy_value_final - energy_value_initial,
        )

        # Save the relaxed and bonded molecules.
        output_molecules_file = molecules_file
        writer = Chem.SDWriter(os.path.join(output_dir, output_molecules_file))
        writer.write(mol, confId=0)


def process_molecules_dir(molecules_dir: str) -> None:
    """Adds bonds and relaxes structures of molecules in a directory."""
    parent_dir = os.path.dirname(molecules_dir)
    bonded_molecules_dir = os.path.join(parent_dir, "bonded_molecules")
    bonded_and_relaxed_molecules_dir = os.path.join(
        parent_dir, "bonded_and_relaxed_molecules"
    )
    os.makedirs(bonded_molecules_dir, exist_ok=True)
    os.makedirs(bonded_and_relaxed_molecules_dir, exist_ok=True)

    logging.info("Adding bonds to molecules in %s", molecules_dir)
    add_bonds_to_molecules(molecules_dir, bonded_molecules_dir)

    logging.info("Relaxing structures of molecules in %s", bonded_molecules_dir)
    relax_structures_of_molecules(
        bonded_molecules_dir, bonded_and_relaxed_molecules_dir
    )


def main(unused_argv: Sequence[str]):
    del unused_argv

    molecules_basedir = FLAGS.molecules_basedir
    molecules_dirs = set()
    for molecules_file in glob.glob(
        os.path.join(molecules_basedir, "**", "*.xyz"), recursive=True
    ):
        molecules_dirs.add(os.path.dirname(molecules_file))

    if not len(molecules_dirs):
        raise ValueError(f"No molecules found in {molecules_basedir}.")

    for molecules_dir in molecules_dirs:
        logging.info("Processing molecules in %s", molecules_dir)
        process_molecules_dir(molecules_dir)


if __name__ == "__main__":
    flags.DEFINE_string("molecules_basedir", None, "Directory containing all molecules.")

    flags.mark_flags_as_required(["molecules_basedir"])
    app.run(main)
