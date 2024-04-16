import ase
import ase.io
import bokeh
from bokeh.io import show
import functools
import jax
import jax.numpy as jnp
import matscipy.neighbours
import numpy as np
import pandas as pd
import pcax
import pickle
import pymatgen
from pymatgen.core import Lattice, Molecule, Structure
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import random
import tqdm
import sys

from analyses.nearest_neighbors import CrystalNN, VoronoiNN
from analyses import metrics
import symphony.models.ptable as ptable

xyzs_path = "../tmqm_data/xyz"
mols_tmqm_ase = []
for mol_file in tqdm.tqdm(os.listdir(xyzs_path)):
    mol_as_ase = ase.io.read(os.path.join(xyzs_path, mol_file), format="xyz")
    if mol_as_ase is None:
        continue
    mols_tmqm_ase.append(mol_as_ase)

def is_transition_metal(z):
    return 2 <= ptable.groups[z-1] <= 11

mols_tmqm = []
for mol in tqdm.tqdm(mols_tmqm_ase):
    struct = Molecule(mol.get_chemical_symbols(), mol.get_positions())
    mols_tmqm.append(struct)


generated_dirs = ["../analyses/analysed_workdirs/tmqmg_apr10_500k/Ni",]
mols_generated = []
for generated_dir in generated_dirs:
    for mol_file in tqdm.tqdm(os.listdir(generated_dir)):
        if mol_file.endswith(".xyz"):
            mol_as_ase = ase.io.read(os.path.join(generated_dir, mol_file), format="xyz")
            if mol_as_ase is None:
                continue
            mols_generated.append(Molecule(mol_as_ase.get_chemical_symbols(), mol_as_ase.get_positions()))

compare = pymatgen.analysis.molecule_structure_comparator.MoleculeStructureComparator()
unique = 0
for m in mols_generated:
    for n in mols_tmqm:
        if not compare.are_equal(m, n):
            unique += 1
print(unique, len(mols_generated))