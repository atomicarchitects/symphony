#!/bin/bash

for l in 0 1 2 3
do
   for c in 32 64 128 256
   do
      #python -m main --config=configs/mace.py --config.max_ell="$l" --config.num_channels="$c" --config.num_interactions=1 --workdir=workdirs/mace/interactions=1/l="$l"/channels="$c"/
      python -m main --config=configs/e3schnet.py --config.max_ell="$l" --config.n_filters="$c" --config.n_atom_basis="$c" --config.n_interactions=1 --workdir=workdirs/e3schnet/interactions=1/l="$l"/channels="$c"/
   done
done
