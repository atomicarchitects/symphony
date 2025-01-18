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
cuda=1
dataset=cath
embedder=nequip
# train=1000
workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/"$dataset"_jan14/e3schnet_and_"$embedder"/$mode/max_targets_$max_targets_per_graph

# CUDA_VISIBLE_DEVICES=$cuda python -m analyses.generate_molecules \
#     --workdir=$workdir \
#     --num_seeds=1 \
#     --num_seeds_per_chunk=1 \
#     --init="N"

CUDA_VISIBLE_DEVICES=$cuda python -m symphony \
    --workdir=$workdir \
    --config=configs/$dataset/e3schnet_and_"$embedder".py \
    --config.num_train_molecules=10 \
    --config.num_val_molecules=1 \
    --config.num_test_molecules=1 \
    --config.shuffle_datasets=False \
    --config.eval_every_steps=5000 \
    --config.generate_every_steps=5000 \
    --config.generation.num_seeds=10 \
    --config.generation.num_seeds_per_chunk=1 \
    --config.generation.posebusters=False \
    --config.num_train_steps=1000000 \
    --config.position_noise_std=0.1 \
    --config.max_num_residues=64 \
    --config.target_distance_noise_std=0.1 \
    --config.max_targets_per_graph=$max_targets_per_graph \
    --config.target_position_predictor.radial_predictor_type="discretized"

    # --config.num_frag_seeds=8 \

    # --config.use_edm_splits=False \