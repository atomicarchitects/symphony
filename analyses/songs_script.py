# Imports
from typing import *
import ase
import ase.db
import ase.io
import ml_collections
import logging
import ase.io
import ase.visualize
logging.getLogger().setLevel(logging.INFO)
import sys
sys.path.append("..")
import analyses.analysis as analysis
import analyses.generate_molecules as generate_molecules
# with ase.db.connect('../qm9_data/qm9gen.db') as conn:
#     for row in conn.select(id=10):
#         mol = row.toatoms()
workdir = 'potato_workdirs/qm9_bessel_embedding_attempt6_edm_splits/e3schnet_and_nequip/interactions=3/l=5/position_channels=2/channels=64'
outputdir = 'analyses/songs_script_output'
beta_species = 1.
beta_position = 1.
step = '960000'
num_seeds = 10
num_seeds_per_chunk = 1
init_molecule = 'analyses/molecules/downloaded/CH3.xyz'  # file name is fine
max_num_atoms = 35
visualize = False
generate_molecules.generate_molecules(
    workdir,
    outputdir,
    beta_species,
    beta_position,
    step,
    num_seeds,
    num_seeds_per_chunk,
    init_molecule,
    max_num_atoms,
    visualize,
)