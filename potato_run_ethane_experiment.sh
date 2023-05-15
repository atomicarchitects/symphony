#!/bin/bash

# Set experiment name
expname=extras/ethane-experiment

# Loop over hyperparameters
for l in 1 2 3 4 5
do
  for c in 32
  do
    for i in 4
    do
      for losstype in "l2" "kl_divergence"
      do
          CUDA_VISIBLE_DEVICES=7 python -m main --config.max_ell="$l" --config.num_channels="$c"  --config.num_interactions="$i" --config.loss_kwargs.position_loss_type="$losstype"  --config=configs/nequip.py --config.train_on_split_smaller_than_chunk=True --config.train_molecules="(6, 7)" --config.val_molecules="(0, 2976)" --config.test_molecules="(0, 2976)" --config.max_n_graphs=8 --config.num_eval_steps=1 --config.eval_every_steps=100 --config.log_every_steps=10 --config.num_eval_steps_at_end_of_training=1 --config.num_train_steps=100000 --config.loss_kwargs.target_position_inverse_temperature=1 --workdir=workdirs/"$expname"/nequip/loss="$losstype"/l="$l"  || break 10
      done
    done
  done
done