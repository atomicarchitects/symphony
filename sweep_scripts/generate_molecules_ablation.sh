#!/bin/sh

workdir="qm9_ablation"

for model in "e3schnet_and_nequip"
do
for l in 1 2 3 4 5
do
for lfocus in 1 2
do
  for i in 3
  do
    for c in 64
    do
      for pc in 4
      do
        for step in best
        do
            CUDA_VISIBLE_DEVICES="$((pc + 3))" python -m analyses.generate_molecules \
            --workdir=workdirs/"$workdir"/"$model"/interactions="$i"/l="$l"/lfocus="$lfocus"/position_channels="$pc"/channels="$c"/  \
            --fait=1 --pit=1 --init=H  \
            --step="$step"  --max_num_atoms=35 --num_seeds=10000 --num_seeds_per_chunk=25   \
            > generation_logs/"$workdir"_l="$l"_lfocus="$lfocus"_pc="$pc"_step="$step".txt  2>&1
        done
      done
    done
  done
done
done
done
