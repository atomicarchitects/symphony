#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -C rocky8
#SBATCH -p sched_mit_tsmidt

cd /home/songk

source .bashrc
conda activate tmqm-dev
cd symphony-tmqm
module load cuda/12.1.0-x86_64

mode=nn
max_targets_per_graph=4
cuda=1
workdir=/pool001/songk/workdirs/tmqmg_coord/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph
#workdir=/pool001/songk/workdirs/tmqmg_feb26/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph

# python -m symphony.data.generation_scripts.tmqm_fragmenter \
#    --mode=nn --max_targets_per_graph=4 --nn_cutoff=3.5 \
#    --output_dir=/pool001/songk/tmqmg_fragments_heavy_first --end_seed=1 \
#    --start_index=25000 --end_index=30000
python -m symphony.run_coordination \
     --config=configs/tmqm/e3schnet_and_nequip.py \
     --config.fragment_logic=$mode \
     --config.max_targets_per_graph=$max_targets_per_graph \
     --config.num_train_steps=100000 \
     --workdir=$workdir
#python -m analyses.generate_molecules \
#    --workdir=$workdir \
#    --max_num_atoms=200 \
#    --num_seeds=50 \
#    --num_seeds_per_chunk=1 \
#    --init=analyses/molecules/downloaded/Ni.xyz \
python -m analyses.evaluate_coord_num --workdir=$workdir
