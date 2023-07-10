#!/bin/bash

# Set experiment name
expname=debug

# Loop over hyperparameters
for nummolecules in 1
do
  for scale_position_logits in True False
  do
    for l in 5
    do
      for c in 32 64
      do
        for i in 4
        do
            CUDA_VISIBLE_DEVICES=4 python -m symphony --config=configs/qm9/nequip-debug.py --config.max_n_graphs=32  --config.max_ell="$l" --config.num_channels="$c"  --config.num_interactions="$i" --config.num_train_steps=100000 --config.train_molecules="(0, $nummolecules)" --config.loss_kwargs.scale_position_logits_by_inverse_temperature="$scale_position_logits"   --workdir=workdirs/"$expname"/nequip/num_molecules="$nummolecules"/scale_position_logits="$scale_position_logits"/interactions="$i"/l="$l"/channels="$c"/  || break 10
        done
      done
    done
  done
done

