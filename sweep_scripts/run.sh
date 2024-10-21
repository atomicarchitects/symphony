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
workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/"$dataset"_multifocus_jul18/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph

CUDA_VISIBLE_DEVICES=$cuda JAX_DEBUG_NANS=True python -m symphony \
    --workdir=$workdir \
    --config=configs/$dataset/e3schnet_and_nequip.py \
    --config.log_every_steps=1000 \
    --config.eval_every_steps=1000 \
    --config.generate_every_steps=1000 \
    --config.generation.num_seeds=1 \
    --config.num_train_steps=100000 \
    --config.fragment_logic=radius \
    --config.max_radius=5.0 \
    --config.generation.num_seeds_per_chunk=2  # TODO figure out how to do this for >1
