#!/bin/sh

workdir="qm9_bessel_embedding_attempt6_edm_splits"

for model in "e3schnet_and_nequip"
do
for l in 5
do
  for i in 3
  do
    for c in 64
    do
      for pc in 2
      do
        for step in 9930000
        do
            CUDA_VISIBLE_DEVICES="$((2))" python -m analyses.generate_molecules --workdir=workdirs/"$workdir"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/ --fait=1 --pit=1 --init=H --step="$step" --max_num_atoms=35 --num_seeds=10000 --num_seeds_per_chunk=25 > "$workdir"_generate_l="$l"_step="$step".txt 2>&1
        done
      done
    done
  done
done
done
