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
import matplotlib.pyplot as plt
import os
import random
import tqdm
import sys

from analyses.nearest_neighbors import CrystalNN, VoronoiNN
from analyses import metrics
import symphony.models.ptable as ptable
sys.path.append('/home/songk/pyspectra/')
import pyspectra
import pyspectra.spectra
import pyspectra.utils
import pyspectra.visualize

sys.path.append('/home/songk/cmap_periodictable/')
from ptable_trends import ptable_plotter

def is_transition_metal(z):
    return 2 <= ptable.groups[z-1] <= 11

def cutoff(mol, site_index, cutoff=3.):
    return mol.get_neighbors(mol[site_index], cutoff)

def cutoff_constructor(c):
    return functools.partial(cutoff, cutoff=c)

def get_chemical_symbols(mol):
    return [el.label for el in mol]

def get_neighbors(mol, site_index, cutoff=3.):
    return [x.label for x in mol.get_neighbors(mol[site_index], cutoff)]

def get_bispectrum_by_el(mol, el, cutoff=3.):
    spectra_comp = pyspectra.spectra.Spectra()
    spectra_comp.set_cutoff(cutoff_constructor(cutoff))
    spectra_comp.load_structure(mol)
    bispectrum = spectra_comp.compute_element_spectra(el)
    return jnp.array(list(bispectrum.values())), list(bispectrum.keys())

def get_bispectra_by_el_pair(mols, center_el, neighbor_el, cutoff=3.):
    bispectra = []
    for mol in tqdm.tqdm(mols):
        if center_el in get_chemical_symbols(mol):
            spectra, sites = get_bispectrum_by_el(mol, center_el, cutoff)
            for s, i in zip(spectra, sites):
                if neighbor_el in get_neighbors(mol, i, cutoff):
                    bispectra.append(s)
    return jnp.array(bispectra)

def get_mmd(center_el, neighbor_el, is_transition_metal=True):
    bispectra_generated = get_bispectra_by_el_pair(mols_generated, center_el, neighbor_el)
    mols_target = mols_tmqm
    if is_transition_metal:
        mols_target = [m for m, _ in mols_by_el[center_el]]
    bispectra_tmqm = get_bispectra_by_el_pair(mols_target, center_el, neighbor_el)

    if len(bispectra_generated) == 0 or len(bispectra_tmqm) == 0:
        return None, None

    key_shuffle, key_rng = jax.random.split(jax.random.PRNGKey(0))
    
    mmd_gen = metrics.compute_maximum_mean_discrepancy(
        bispectra_generated,
        bispectra_tmqm,
        rng=key_rng,
        batch_size=50,
        num_batches=10
    )
    mmd_tmqm = metrics.compute_maximum_mean_discrepancy(
        jax.random.permutation(key_shuffle, bispectra_tmqm)[:500],
        bispectra_tmqm,
        rng=key_rng,
        batch_size=50,
        num_batches=10
    )

    return mmd_gen, mmd_tmqm

# get data
print("Getting TMQM data...")
xyzs_path = "../tmqm_data/xyz"
mols_tmqm_ase = []
for mol_file in tqdm.tqdm(os.listdir(xyzs_path)):
    mol_as_ase = ase.io.read(os.path.join(xyzs_path, mol_file), format="xyz")
    if mol_as_ase is None:
        continue
    mols_tmqm_ase.append(mol_as_ase)

mols_tmqm = []
for mol in tqdm.tqdm(mols_tmqm_ase):
    struct = Molecule(mol.get_chemical_symbols(), mol.get_positions())
    mols_tmqm.append(struct)

with open("mols_by_el.pkl", "rb") as f:
    mols_by_el = pickle.load(f)

print("Getting generated data...")
generated_dir = "../analyses/analysed_workdirs/tmqmg_apr10_500k/Ni"
mols_generated = []
for mol_file in tqdm.tqdm(os.listdir(generated_dir)):
    if mol_file.endswith(".xyz"):
        mol_as_ase = ase.io.read(os.path.join(generated_dir, mol_file), format="xyz")
        if mol_as_ase is None:
            continue
        mols_generated.append(Molecule(mol_as_ase.get_chemical_symbols(), mol_as_ase.get_positions()))

df_dict = {}
print("Getting MMD scores...")
# for neighbor_el in ['B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I']:
#     print("Getting MMD scores for Ni and", neighbor_el)
#     mmd_gen, mmd_target = get_mmd('Ni', neighbor_el, True)
#     if mmd_gen is None or mmd_target is None:
#         continue
#     with open(f"spectra/bispectra_Ni_{neighbor_el}.pkl", "wb") as f:
#         pickle.dump((mmd_gen, mmd_target), f)
#     df_dict['center_el'] = 'Ni'
#     df_dict['neighbor_el'] = neighbor_el
#     df_dict['mmd_gen'] = mmd_gen
#     df_dict['mmd_target'] = mmd_target
#     print(('Ni', neighbor_el, mmd_gen, mmd_target))

for center_el in ['C', 'N', 'O']:
    for neighbor_el in ['H', 'C', 'N', 'O',]:
        print("Getting MMD scores for", center_el, "and", neighbor_el)
        mmd_gen, mmd_target = get_mmd(center_el, neighbor_el, False)
        if mmd_gen is None or mmd_target is None:
            continue
        # with open(f"spectra/bispectra_{center_el}_{neighbor_el}.pkl", "wb") as f:
        #     pickle.dump((mmd_gen, mmd_target), f)
        df_dict['center_el'] = center_el
        df_dict['neighbor_el'] = neighbor_el
        df_dict['mmd_gen'] = mmd_gen
        df_dict['mmd_target'] = mmd_target
        print((center_el, neighbor_el, mmd_gen, mmd_target))

# df = pd.DataFrame(df_dict)
# df.to_csv("mmd.csv", index=False)