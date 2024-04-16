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
sys.path.append('/home/songk/pyspectra/')
import pyspectra
import pyspectra.spectra
import pyspectra.utils
import pyspectra.visualize

sys.path.append('/home/songk/cmap_periodictable/')
from ptable_trends import ptable_plotter


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

generated_dirs = ["../analyses/analysed_workdirs/tmqmg_apr10_500k/Ni", "../analyses/analysed_workdirs/tmqmg_apr10_500k/W"]
mols_generated = []
for generated_dir in generated_dirs:
    for mol_file in tqdm.tqdm(os.listdir(generated_dir)):
        if mol_file.endswith(".xyz"):
            mol_as_ase = ase.io.read(os.path.join(generated_dir, mol_file), format="xyz")
            if mol_as_ase is None:
                continue
            mols_generated.append(Molecule(mol_as_ase.get_chemical_symbols(), mol_as_ase.get_positions()))

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
    for mol in mols:
        if center_el in get_chemical_symbols(mol):
            spectra, sites = get_bispectrum_by_el(mol, center_el, cutoff)
            for s, i in zip(spectra, sites):
                if neighbor_el in get_neighbors(mol, i, cutoff):
                    bispectra.append(s)
    return jnp.array(bispectra)

def get_bispectra_by_el_pair(mols, center_el, neighbor_el, cutoff=3.):
    bispectra = []
    for mol in mols:
        if center_el in get_chemical_symbols(mol):
            spectra, sites = get_bispectrum_by_el(mol, center_el, cutoff)
            for s, i in zip(spectra, sites):
                if neighbor_el in get_neighbors(mol, i, cutoff):
                    bispectra.append(s)
    return jnp.array(bispectra)

def get_bispectra_by_el_env(mols, center_el, neighbor_els, cutoff=3.):
    bispectra = []
    for mol in mols:
        if center_el in get_chemical_symbols(mol):
            spectra, sites = get_bispectrum_by_el(mol, center_el, cutoff)
            for s, i in zip(spectra, sites):
                if set(neighbor_els) == set(get_neighbors(mol, i, cutoff)):
                    bispectra.append(s)
    return jnp.array(bispectra)

def get_graphs(center_el, neighbor_els):
    fn = get_bispectra_by_el_env
    if type(neighbor_els) == str:
        fn = get_bispectra_by_el_pair
    bispectra_generated = fn(mols_generated, center_el, neighbor_els)
    bispectra_tmqm = []
    for i in tqdm.tqdm(jax.random.permutation(jax.random.PRNGKey(0), jnp.arange(len(mols_tmqm)))[:1000]):
        bispectra_tmqm.append(fn([mols_tmqm[i]], center_el, neighbor_els))
    bispectra_tmqm = jnp.concatenate([x for x in bispectra_tmqm if x.shape != (0,)], axis=0)


    bispectra_tmqm2 = []
    for i in tqdm.tqdm(jax.random.permutation(jax.random.PRNGKey(1), jnp.arange(len(mols_tmqm)))[:1000]):
        bispectra_tmqm2.append(fn([mols_tmqm[i]], center_el, neighbor_els))
    bispectra_tmqm2 = jnp.concatenate([x for x in bispectra_tmqm2 if x.shape != (0,)], axis=0)

    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    mmd_gen = metrics.compute_maximum_mean_discrepancy(
        bispectra_generated,
        bispectra_tmqm,
        key1,
        batch_size=50,
        num_batches=20
    )
    mmd_tmqm = metrics.compute_maximum_mean_discrepancy(
        bispectra_tmqm,
        bispectra_tmqm2,
        rng=key2,
        batch_size=50,
        num_batches=20
    )
    print(f"Num of bispectra: {len(bispectra_generated)}, {len(bispectra_tmqm)}")
    print(f"MMD for {center_el}, {neighbor_els}: {mmd_gen}, {mmd_tmqm}")

    key_1, key_2 = jax.random.split(jax.random.PRNGKey(0))

    n_samples = 20
    sampled_bispectra_generated = jax.random.choice(key_1, bispectra_generated, shape=(n_samples,), replace=False)
    sampled_bispectra_tmqm = jax.random.choice(key_2, bispectra_tmqm, shape=(n_samples,), replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    vmin = min(sampled_bispectra_generated.min(), sampled_bispectra_tmqm.min())
    vmax = max(sampled_bispectra_generated.max(), sampled_bispectra_tmqm.max())
    ax[0].imshow(
        sampled_bispectra_generated, cmap="PRGn", vmin=vmin, vmax=vmax
    )
    ax[0].set_xlabel("Symphony")
    ax[0].set_ylabel("Local Environment #")

    im = ax[1].imshow(
        sampled_bispectra_tmqm, cmap="PRGn", vmin=vmin, vmax=vmax
    )
    ax[1].set_xlabel("TMQM")
    ax[1].set_ylabel("Local Environment #")

    fig.colorbar(im, ax=ax[1])
    fig.suptitle(f"{center_el}-{neighbor_els} Bispectra")

    fig.show()
    fig.savefig(f"plots/tmqm_bispectra/{center_el}_{neighbor_els}_bispectra.png")

get_graphs('C', ['C', 'H'])
get_graphs('C', ['C', 'H', 'O'])
get_graphs('Ni', 'C')
get_graphs('Ni', 'N')
get_graphs('Ni', 'O')