#!/bin/bash

# Set experiment name
expname=v7

# Loop over hyperparameters
for model in "nequip"
do
  for l in 1 2 3 4 5
  do
    for c in 32
    do
      for i in 4
      do
          CUDA_VISIBLE_DEVICES="$l" python -m main --config=configs/"$model".py --config.max_n_graphs=32  --config.max_ell="$l" --config.num_channels="$c"  --config.num_interactions="$i" --config.num_train_steps=1000000 --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/channels="$c"/  > "$expname"_model="$model"_l="$l"_c="$c"_i="$i".txt 2>&1  &
      done
    done
  done
done


