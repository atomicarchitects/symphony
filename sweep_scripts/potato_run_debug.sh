#!/bin/bash

# Set experiment name
expname=debug

# Loop over hyperparameters
for nummolecules in 1
do
  for model in nequip
  do
      for l in 5
      do
        for c in 32
        do
          for i in 4
          do
            CUDA_VISIBLE_DEVICES=6 python -m main --config=configs/"$model".py  --config.train_on_split_smaller_than_chunk=True  --config.max_n_graphs=8  --config.max_ell="$l" --config.num_channels="$c"  --config.num_interactions="$i" --config.num_train_steps=30000 --config.train_molecules="(0, $nummolecules)"  --workdir=workdirs/"$expname"/"$model"/num_molecules="$nummolecules"/interactions="$i"/l="$l"/channels="$c"/  || break 10
          done
        done
      done
  done
done

