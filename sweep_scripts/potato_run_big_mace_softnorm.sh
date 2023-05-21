#!/bin/bash

# Set experiment name
expname=v5

# Loop over hyperparameters
for l in 5
do
  for c in 32
  do
    for i in 4
    do
        CUDA_VISIBLE_DEVICES=5 python -m main --config=configs/mace.py     --config.max_ell="$l" --config.num_channels="$c"  --config.num_interactions="$i" --workdir=workdirs/"$expname"/mace-softnorm/interactions="$i"/l="$l"/channels="$c"/  || break 10
    done
  done
done

