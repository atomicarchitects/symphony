#!/bin/bash

# Set experiment name
expname=extras/sample_complexity

# Loop over hyperparameters
for nummoleculesfactor in 1 2 4 8 16
do
  nummolecules=$(( 2976*nummoleculesfactor ))

  for l in 0 1 2 3 4 5
  do
    for c in 32
    do
      for i in 4
      do
        CUDA_VISIBLE_DEVICES=5 python -m main --config=configs/mace.py     --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --config.train_molecules="(0,$nummolecules)" --workdir=workdirs/"$expname"/mace/nummoleculesfactor="$nummoleculesfactor"/interactions="$i"/l="$l"/channels="$c"/     || break 10
        CUDA_VISIBLE_DEVICES=5 python -m main --config=configs/e3schnet.py --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --config.train_molecules="(0,$nummolecules)" --workdir=workdirs/"$expname"/e3schnet//nummoleculesfactor="$nummoleculesfactor"/interactions="$i"/l="$l"/channels="$c"/ || break 10
      done
    done
  done
done

