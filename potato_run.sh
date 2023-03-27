#!/bin/bash

# Set experiment name
expname=v4

# Make directories
mkdir -p workdirs/"$expname"/mace
mkdir -p workdirs/"$expname"/e3schnet

# Loop over hyperparameters
for l in 0 1 2 3 4 5
do
  for c in 32
  do
    for i in 1 2 3 4 5 6 7 8 9 10
    do
      if [ $scriptnum -eq 1 ]
      then
        if [ $(( $i % 2 )) -eq 0 ]
        then
          CUDA_VISIBLE_DEVICES=5 python -m main --config=configs/mace.py     --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/"$expname"/mace/interactions="$i"/l="$l"/channels="$c"/   
        fi
  
      elif [ $scriptnum -eq 2 ]
      then
        if [ $(( $i % 2 )) -eq 1 ]
        then
          CUDA_VISIBLE_DEVICES=6 python -m main --config=configs/mace.py     --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/"$expname"/mace/interactions="$i"/l="$l"/channels="$c"/   
        fi
  
      elif [ $scriptnum -eq 3 ]
      then
        if [ $(( $i % 2 )) -eq 0 ]
        then
          CUDA_VISIBLE_DEVICES=7 python -m main --config=configs/e3schnet.py --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/"$expname"/e3schnet/interactions="$i"/l="$l"/channels="$c"/
        fi

      elif [ $scriptnum -eq 4 ]
      then
        if [ $(( $i % 2 )) -eq 1 ]
        then
          CUDA_VISIBLE_DEVICES=4 python -m main --config=configs/e3schnet.py --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/"$expname"/e3schnet/interactions="$i"/l="$l"/channels="$c"/
        fi

      fi
    done
  done
done

