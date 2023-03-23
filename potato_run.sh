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
        grun python -m main --config=configs/mace.py     --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/v"$vnum"/mace/interactions="$i"/l="$l"/channels="$c"/     1>> 2>> workdirs/v"$vnum"/mace/out.txt &
        sleep 10
        grun python -m main --config=configs/e3schnet.py --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/v"$vnum"/e3schnet/interactions="$i"/l="$l"/channels="$c"/ 1>> 2>> workdirs/v"$vnum"/e3schnet/out.txt &
        sleep 10
        grun python -m main --config=configs/nequip.py   --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/v"$vnum"/nequip/interactions="$i"/l="$l"/channels="$c"/   1>> 2>> workdirs/v"$vnum"/nequip/out.txt &
        sleep 10
    done
  done
done

