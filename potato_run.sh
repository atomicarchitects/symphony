#!/bin/bash

for l in 0 1 2 3 4 5
do
   for c in 64
   do
      for i in 1 2 3 4
      do
        python -m main --config=configs/mace.py     --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/v2/mace/interactions="$i"/l="$l"/channels="$c"/
        python -m main --config=configs/e3schnet.py --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions="$i" --workdir=workdirs/v2/e3schnet/interactions="$i"/l="$l"/channels="$c"/
      done
    done
done
