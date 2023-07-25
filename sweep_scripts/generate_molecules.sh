#!/bin/sh

workdir="qm9_15JUL"

for l in 1 2 3 4
do
    for i in 3
    do
      for pc in 2
      do
          CUDA_VISIBLE_DEVICES=6 python -m analyses.generate_molecules  --workdir=workdirs/"$workdir"/nequip/interactions="$i"/l="$l"/position_channels="$pc"/channels=64/  --fait=1 --pit=1 --init=H  --max_num_atoms=30 --num_seeds=1000 --num_seeds_per_chunk=25
      done
    done
done
