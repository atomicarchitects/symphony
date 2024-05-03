#!/bin/bash
#SBATCH --account=bbyc-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH -c 8
#SBATCH --time=2-00:00:00

cd /u/skim12
source .bashrc
module load gcc-runtime/11.4.0
module load cuda/12.4.0
module load python/3.10.13
source .venv/rdkit/bin/activate

cd symphony

dataset=tmqm
workdir=/u/skim12/workdirs/"$dataset"_may1/e3schnet_and_nequip/nn/max_targets_4

python -m analyses.generate_molecules \
    --workdir=$workdir \
    --max_num_atoms=200 \
    --init=Ni \
    --num_seeds=100 \
    --num_seeds_per_chunk=2
