#!/bin/sh

for max_num_atoms in 5 10 20 30
do
    for num_seeds in 25 100 500 1000 5000 10000
    do
        CUDA_VISIBLE_DEVICES=0 time python -m analyses.generate_molecules_batched --workdir=workdirs/qm9_10JUL/nequip/interactions=2/l=4/position_channels=5/channels=64/ --fait=1 --pit=1 --init=H --max_num_atoms="$max_num_atoms" --num_seeds="$num_seeds" --num_seeds_per_chunk=25 > benchmark_generate_batched/benchmark_generate_batched_atoms="$max_num_atoms"_seeds="$num_seeds".txt 2>&1
    done
done
