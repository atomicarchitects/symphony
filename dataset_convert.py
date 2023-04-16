import sys

sys.path.append('analyses')

import analysis

import input_pipeline

config, _, _, _ = analysis.load_from_workdir('/home/ameyad/spherical-harmonic-net/workdirs/v3/mace/interactions=4/l=5/channels=32')

print("config loaded")

input_pipeline.dataset_as_database(config, 'train', 'qm9_data/qm9-train.db')
input_pipeline.dataset_as_database(config, 'all', 'qm9_data/qm9-all.db')
