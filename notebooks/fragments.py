import ase
import ase.io
import bokeh
from bokeh.io import show
import functools
import matscipy.neighbours
import numpy as np
import pandas as pd
import symphony.models.ptable as ptable
import matplotlib.pyplot as plt
import os
import tqdm
import sys
sys.path.append('/home/songk/cmap_periodictable/')
from ptable_trends import ptable_plotter


xyzs_path = "../tmqm_data/tmqm/data/xyz"
mols = []
for mol_file in tqdm.tqdm(os.listdir(xyzs_path)):
    mol_as_ase = ase.io.read(os.path.join(xyzs_path, mol_file), format="xyz")
    if mol_as_ase is None:
        continue
    mols.append(mol_as_ase)

element_counts = {}
element_counts_per_mol = {}
for mol in tqdm.tqdm(mols[:10]):
    e_to_add = set()
    for symbol, num in zip(mol.get_chemical_symbols(), mol.get_atomic_numbers()):
        if ptable.groups[num-1] < 2 or ptable.groups[num-1] > 11: continue
        if symbol not in element_counts:
            element_counts[symbol] = 0
            element_counts_per_mol[symbol] = 0
        element_counts[symbol] += 1
        e_to_add.add(symbol)
    for symbol in e_to_add:
        element_counts_per_mol[symbol] += 1

element_count_df = pd.DataFrame(columns=["element", "count"], data=[[k, v] for k, v in element_counts.items()])
element_count_df.to_csv("element_counts.csv", index=False, header=False)
fig = ptable_plotter("element_counts.csv", log_scale=True)
bokeh.io.save(fig, 'element_counts.html')

element_count_per_mol_df = pd.DataFrame(columns=["element", "count"], data=[[k, v] for k, v in element_counts_per_mol.items()])
element_count_per_mol_df.to_csv("element_counts_per_mol.csv", index=False, header=False)
fig = ptable_plotter("element_counts_per_mol.csv", log_scale=True)
bokeh.io.save(fig, 'element_counts_per_mol.html')