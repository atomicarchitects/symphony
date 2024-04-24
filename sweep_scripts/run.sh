#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -C rocky8
#SBATCH -p sched_mit_tsmidt

#cd /home/songk

#source .bashrc
#conda activate tmqm-dev
#cd symphony-tmqm
#module load cuda/12.1.0-x86_64

mode=nn
max_targets_per_graph=4
cuda=4
dataset=qm9
#workdir=/pool001/songk/workdirs/tmqmg_coord/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph
#workdir=/pool001/songk/workdirs/tmqmg_feb26/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph
workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/$dataset_apr22/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph

# python -m symphony.data.generation_scripts.tmqm_fragmenter \
#    --mode=nn --max_targets_per_graph=4 --nn_cutoff=3.5 \
#    --output_dir=/pool001/songk/tmqmg_fragments_heavy_first --end_seed=1 \
#    --start_index=25000 --end_index=30000
# CUDA_VISIBLE_DEVICES=$cuda python -m symphony.data.generation_scripts.tmqm_ni_fragmenter \
#    --mode=$mode --max_targets_per_graph=$max_targets_per_graph --nn_cutoff=3.5 \
#    --output_dir=/data/NFS/potato/songk/tmqm_ni_fragments --end_seed=1 \
#     --end_index=1 --chunk=1

CUDA_VISIBLE_DEVICES=$cuda python -m symphony \
    --config=configs/$dataset/e3schnet_and_nequip.py \
    --workdir=$workdir

# CUDA_VISIBLE_DEVICES=$cuda python -m analyses.generate_molecules_intermediates \
#     --workdir=$workdir \
#     --max_num_steps=200 \
#     --num_seeds=5 \
#     --init=analyses/molecules/downloaded/Ni.xyz
#python -m analyses.evaluate_coord_num --workdir=$workdir
