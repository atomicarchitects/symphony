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
max_targets_per_graph=1
cuda=0
dataset=qm9
embedder=nequip
# train=1000
workdir=/data/NFS/radish/songk/spherical-harmonic-net/workdirs/"$dataset"_dec31/e3schnet_and_"$embedder"/$mode/max_targets_$max_targets_per_graph
# workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/"$dataset"_nov18_"$train"/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph

# CUDA_VISIBLE_DEVICES=$cuda python -m analyses.generate_molecules \
#     --workdir=$workdir
    # --num_seeds=50 \
    # --num_seeds_per_chunk=1 \
    # --max_num_atoms=200 \
    # --init=analyses/molecules/downloaded/Ni.xyz

CUDA_VISIBLE_DEVICES=$cuda python -m symphony \
    --workdir=$workdir \
    --config=configs/$dataset/e3schnet_and_"$embedder".py \
    --config.eval_every_steps=5000 \
    --config.generate_every_steps=5000 \
    --config.num_train_steps=1000000 \
    --config.position_noise_std=0.1 \
    --config.target_distance_noise_std=0.1 \
    --config.max_targets_per_graph=$max_targets_per_graph


    # --config.num_train_molecules=1000 \
    # --config.num_val_molecules=1 \
    # --config.num_test_molecules=1 \
    # --config.use_edm_splits=False \
    # --config.shuffle_datasets=False \