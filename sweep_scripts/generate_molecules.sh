#!/bin/sh

workdir="qm9_bessel_embedding"

for model in "nequip"
do
for l in 1 2 3
do
  for i in 3
  do
    for c in 64
    do
      for pc in 2
      do
        for step in best
        do
            CUDA_VISIBLE_DEVICES="$((l))" python -m analyses.generate_molecules  --workdir=workdirs/"$workdir"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/global_embed=False  --fait=1 --pit=1 --init=C  --step="$step"  --max_num_atoms=30 --num_seeds=10000 --num_seeds_per_chunk=25 > "$workdir"_generate_l="$l".txt  2>&1  &
        done
      done
    done
  done
done
done
