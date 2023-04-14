import sys

sys.path.append('analyses')

import analysis

import input_pipeline

config, _, _, _ = analysis.load_from_workdir('/home/ameyad/spherical-harmonic-net/workdirs/v3/mace/interactions=4/l=5/channels=32')

config.test_molecules = (53568, 133885)

input_pipeline.dataset_as_database(config, 'qm9_data/qm9-train.db')
