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
cuda=0,1
dataset=tmqm
workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/"$dataset"_old/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph


CUDA_VISIBLE_DEVICES=$cuda python -m symphony \
    --config=configs/$dataset/e3schnet_and_nequip.py \
    --workdir=$workdir

# CUDA_VISIBLE_DEVICES=$cuda python -m analyses.generate_molecules_intermediates \
#     --workdir=$workdir \
#     --max_num_steps=200 \
#     --num_seeds=5 \
#     --init=analyses/molecules/downloaded/Ni.xyz
#python -m analyses.evaluate_coord_num --workdir=$workdir
