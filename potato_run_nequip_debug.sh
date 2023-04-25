#!/bin/bash

# Set experiment name
expname=debug

# Loop over hyperparameters
for nummolecules in 1 2 5 10
  do
    for l in 5
    do
      for c in 32
      do
        for i in 4
        do
            CUDA_VISIBLE_DEVICES=4 python -m main --config=configs/nequip-debug.py --config.max_n_graphs=32  --config.max_ell="$l" --config.num_channels="$c"  --config.num_interactions="$i" --config.num_train_steps=100000 --config.train_molecules="(0, $nummolecules)"  --workdir=workdirs/"$expname"/nequip/num_molecules="$nummolecules"/interactions="$i"/l="$l"/channels="$c"/  || break 10
        done
      done
    done
done

