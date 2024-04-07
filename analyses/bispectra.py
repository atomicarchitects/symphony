import ase
import ase.io
import bokeh
from bokeh.io import show
import functools
import matscipy.neighbours
import numpy as np
import pandas as pd
import pcax
import pickle
import pymatgen
from pymatgen.core import Lattice, Molecule, Structure
import matplotlib.pyplot as plt
import os
import tqdm
import sys

import symphony.models.ptable as ptable
sys.path.append('/home/songk/pyspectra/')
import pyspectra
import pyspectra.spectra
import pyspectra.utils
import pyspectra.visualize

sys.path.append('/home/songk/cmap_periodictable/')
from ptable_trends import ptable_plotter

with open('notebooks/spectra_by_el.pkl', 'rb') as f:
    spectra_by_el = pickle.load(f)

cos_sim = {}
els = list(spectra_by_el.keys())

for i in range(len(spectra_by_el)):
    spectra_by_el[els[i]] = [x for x in spectra_by_el[els[i]] if x is not None]
    # try:
    #     a = np.array(spectra_by_el[els[i]])
    # except Exception as e:
    #     print([x.shape for x in spectra_by_el[els[i]]])
    #     raise e
for i in range(len(spectra_by_el)):
    print(f"Comparing {els[i]}")
    for j in tqdm.tqdm(range(i, len(spectra_by_el))):
        spectra1 = np.array(spectra_by_el[els[i]])
        spectra2 = np.array(spectra_by_el[els[j]])
        norm1 = np.repeat(np.linalg.norm(spectra1, axis=-1).reshape(-1, 1), spectra2.shape[0], axis=-1)
        norm2 = np.repeat(np.linalg.norm(spectra2, axis=-1).reshape(-1, 1), spectra1.shape[0], axis=-1).T
        cos = spectra1 @ spectra2.T / norm1 / norm2
        if i == j:
            cos_sim[(els[i], els[j])] = (np.sum(cos) - spectra1.shape[0]) / (spectra1.shape[0] * (spectra1.shape[0] - 1))
        else:
            cos_sim[(els[i], els[j])] = np.sum(cos) / (spectra1.shape[0] * spectra2.shape[0])
        #cos_sim[(els[0], els[1])] = l
            
print(cos_sim)

with open("notebooks/cosine_similarity.pkl", "wb") as f:
    pickle.dump(cos_sim, f)
