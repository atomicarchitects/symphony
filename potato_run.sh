#!/bin/bash

# Set experiment version number
vnum=3

# Make directories
mkdir -p workdirs/v"$vnum"/mace
mkdir -p workdirs/v"$vnum"/e3schnet
mkdir -p workdirs/v"$vnum"/nequip

# Loop over hyperparameters
for l in 0 1 2 3
do
  for c in 32
  do
    for i in 1 2 3 4
    do
      if [ $scriptnum -eq 1 ]
      then
        CUDA_VISIBLE_DEVICES=6 python -m main --config=configs/mace.py     --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/v"$vnum"/mace/interactions="$i"/l="$l"/channels="$c"/     >> workdirs/v"$vnum"/mace/out.txt     2>&1
      elif [ $scriptnum -eq 2 ]
      then
        CUDA_VISIBLE_DEVICES=7 python -m main --config=configs/e3schnet.py --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/v"$vnum"/e3schnet/interactions="$i"/l="$l"/channels="$c"/ >> workdirs/v"$vnum"/e3schnet/out.txt 2>&1
      elif [ $scriptnum -eq 3 ]
      then
        CUDA_VISIBLE_DEVICES=1 python -m main --config=configs/nequip.py   --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/v"$vnum"/nequip/interactions="$i"/l="$l"/channels="$c"/   >> workdirs/v"$vnum"/nequip/out.txt   2>&1
    done
  done
done

