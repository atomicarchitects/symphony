#!/bin/bash
#SBATCH --account=bbyc-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH -c 8
#SBATCH --time=1-00:00:00
#SBATCH -J symphony-periodic-supercell

cd /scratch/bbyc/skim12
source .bashrc
module load gcc-runtime/11.4.0
module load cuda/12.4.0
module load python/3.10.13
source .venv/rdkit/bin/activate

cd symphony-periodic

dataset=perov5
max_targets=1
workdir=/scratch/bbyc/skim12/workdirs/"$dataset"_supercell_may8/e3schnet_and_nequip/nn/max_targets_$max_targets

python -m symphony \
    --config=configs/"$dataset"/e3schnet_and_nequip.py \
    --workdir=$workdir \
    --use_wandb=False \
    --config.supercell=True \
    --config.max_targets_per_graph=$max_targets
