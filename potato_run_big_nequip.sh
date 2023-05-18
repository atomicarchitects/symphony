#!/bin/bash

# Set experiment name
expname=v6

# Loop over hyperparameters
for l in 5
do
  for c in 128
  do
    for i in 4
    do
        CUDA_VISIBLE_DEVICES=7 python -m main --config=configs/nequip.py --config.max_n_graphs=32  --config.max_ell="$l" --config.num_channels="$c"  --config.num_interactions="$i" --config.loss_kwargs.position_loss_type="l2" --config.num_train_steps=3200000 --config.loss_kwargs.target_position_inverse_temperature=1 --workdir=workdirs/"$expname"/nequip-l2/interactions="$i"/l="$l"/channels="$c"/  || break 10
    done
  done
done


