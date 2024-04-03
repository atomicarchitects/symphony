#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p sched_mit_hill

cd /home/songk

source .bashrc
conda activate tmqm-dev
cd symphony-tmqm

mode=nn
max_targets_per_graph=4
cuda=1

python -m symphony.data.generation_scripts.tmqm_fragmenter \
   --mode=nn --max_targets_per_graph=4 --nn_cutoff=3.5 \
   --output_dir=/pool001/songk/tmqmg_fragments_heavy_first --end_seed=1 \
   --start_index=60000 --end_index=63000

# CUDA_VISIBLE_DEVICES=0 python -m symphony.data.generation_scripts.tmqm_ni_fragmenter \
#    --mode=nn --max_targets_per_graph=4 --nn_cutoff=3.5 \
#    --output_dir=/data/NFS/potato/songk/tmqm_ni_fragments_single --end_seed=8 \
#     --end_index=1 --chunk=1