#!/bin/bash

# Set experiment name
expname=v7

# Loop over hyperparameters
for model in "nequip"
do
  for l in 1
  do
    for c in 64
    do
      for i in 1 2 3 4 5 6
      do
          CUDA_VISIBLE_DEVICES="$((i-1))" python -m main --config=configs/qm9/"$model".py --config.max_n_graphs=32  --config.max_ell="$l" --config.num_channels="$c"  --config.num_interactions="$i" --config.num_train_steps=200000 --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/channels="$c"/  > "$expname"_model="$model"_l="$l"_c="$c"_i="$i".txt 2>&1  &
      done
    done
  done
done


