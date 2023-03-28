#!/bin/bash

# Set experiment name
expname=extras/num_layers

# Loop over hyperparameters
for numlayers in 1 2 3 4
do
  for l in 5
  do
    for c in 32
    do
      for i in 4
      do
        CUDA_VISIBLE_DEVICES=4 python -m main --config=configs/mace.py      --config.focus_predictor.num_layers="$numlayers" --config.target_species_predictor.num_layers="$numlayers" --config.target_position_predictor.num_layers="$numlayers"  --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/"$expname"/mace/interactions="$i"/l="$l"/channels="$c"/num_layers="$numlayers"/      || break 10
        CUDA_VISIBLE_DEVICES=4 python -m main --config=configs/e3schnet.py  --config.focus_predictor.num_layers="$numlayers" --config.target_species_predictor.num_layers="$numlayers" --config.target_position_predictor.num_layers="$numlayers"  --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/"$expname"/e3schnet/interactions="$i"/l="$l"/channels="$c"/num_layers="$numlayers"/  || break 10
      done
    done
  done
done

