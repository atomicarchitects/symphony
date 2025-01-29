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
cuda=0,1
dataset=miniprotein
embedder=nequip
# train=1000
workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/"$dataset"_jan28_edge_cutoff_7/e3schnet_and_"$embedder"/$mode/max_targets_$max_targets_per_graph

# CUDA_VISIBLE_DEVICES=$cuda python -m analyses.generate_molecules \
#     --workdir=$workdir \
#     --num_seeds=20 \
#     --num_seeds_per_chunk=1 \
#     --dataset=$dataset \
#     --step=100000 \
#     --init="N"

CUDA_VISIBLE_DEVICES=$cuda python -m symphony \
    --workdir=$workdir \
    --config=configs/$dataset/e3schnet_and_"$embedder".py \
    --config.eval_every_steps=5000 \
    --config.generate_every_steps=5000 \
    --config.generation.start_seed=101 \
    --config.generation.num_seeds=20 \
    --config.generation.num_seeds_per_chunk=1 \
    --config.generation.posebusters=False \
    --config.num_train_steps=1000000 \
    --config.position_noise_std=0.1 \
    --config.max_num_residues=65 \
    --config.generation.init_molecules=analyses/molecules/downloaded/N.pdb \
    --config.target_distance_noise_std=0.1 \
    --config.num_frag_seeds=8 \
    --config.radial_cutoff=7.0 \
    --config.max_targets_per_graph=$max_targets_per_graph \
    --config.target_position_predictor.radial_predictor_type="discretized"

    # --config.focus_and_target_species_predictor.k=5 \

    # --config.alpha_carbons_only=True \
    # --config.target_position_predictor.radial_predictor.min_radius=2.0 \
    # --config.target_position_predictor.radial_predictor.max_radius=8.0 \

    # --config.use_edm_splits=False \