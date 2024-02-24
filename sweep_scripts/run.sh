#!/bin/bash
#SBATCH -N 1
#SBATCH -C rocky8
#SBATCH -p sched_mit_hill
#SBATCH --gres=gpu:1
#SBATCH -o slurm_logs/tmqmg_train

cd /home/songk

source .bashrc
conda activate tmqm-dev
cd symphony-tmqm

mode=nn
max_targets_per_graph=4
cuda=1
workdir=/home/songk/data/workdirs/tmqmg_fragments_reduced_elementinfo/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph

#python -m symphony.data.generation_scripts.tmqm_fragmenter \
#    --mode=nn --max_targets_per_graph=4 --nn_cutoff=3.5 \
#    --output_dir=/home/songk/data/tmqmg_fragments_heavy_first --end_seed=1 \
#    --start_index=8000 --end_index=10000
CUDA_VISIBLE_DEVICES=0 python -m symphony.run_coordination \
     --config=configs/tmqm/e3schnet_and_nequip.py \
     --config.fragment_logic=$mode \
     --config.max_targets_per_graph=$max_targets_per_graph \
     --config.num_train_steps=100000 \
     --workdir=$workdir
# CUDA_VISIBLE_DEVICES=$cuda python -m analyses.generate_molecules \
#     --workdir=$workdir \
#     --max_num_atoms=200 \
#     --num_seeds=50 \
#     --num_seeds_per_chunk=1 \
#     --init=analyses/molecules/downloaded/Ni.xyz \
# CUDA_VISIBLE_DEVICES=$cuda python -m analyses.conditional_generation --workdir=$workdir
