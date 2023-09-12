#!/bin/sh

workdir="qm9_bessel_embedding"

for l in 4 5
do
  for i in 3
  do
    for c in 64
    do
      for pc in 2
      do
        for step in best
        do
            CUDA_VISIBLE_DEVICES="$((l+2))" python -m analyses.generate_molecules  --workdir=workdirs/"$workdir"/nequip/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/global_embed=False  --fait=1 --pit=1 --init=H  --step="$step"  --max_num_atoms=35 --num_seeds=10000 --num_seeds_per_chunk=25 > "$workdir"_l="$l"_generate.txt 2>&1 &
        done
      done
    done
  done
done
