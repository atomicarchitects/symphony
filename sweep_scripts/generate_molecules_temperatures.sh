#!/bin/sh

workdir="qm9_bessel_embedding_attempt6_edm_splits_iclr2024_submission"

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
        for ra in 359
        do
        for rb in 180
        do
        for fait in 0.5 0.75 1 1.5 2
        do
        for pit in 0.5 0.75 1 1.5 2
        do
            CUDA_VISIBLE_DEVICES=6 python -m analyses.generate_molecules \
            --workdir=workdirs/"$workdir"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/ \
            --fait="$fait" --pit="$pit" --res_beta="$rb" --res_alpha="$ra" --init=H  --step="$step" \
            --max_num_atoms=35 --num_seeds=10000 --num_seeds_per_chunk=25  \
            > generation_logs/temperatures/"$workdir"_generate_l="$l"_step="$step"_resbeta="$rb"_resalpha="$ra"_fait="$fait"_pit="$pit".txt  2>&1
        done
        done
      done
    done
  done
done
done
done
done
done
