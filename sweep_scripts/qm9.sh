#!/bin/bash

# Set experiment name
dataset="qm9"
expname="$dataset"_bessel_embedding

# Loop over hyperparameters
for model in "nequip"
do
  for l in 5
  do
    for pc in 2
    do
      for c in 64
      do
        for i in 3
	do
          for globalembed in False
          do
              CUDA_VISIBLE_DEVICES="$((l+1))" python -m symphony --config=configs/"$dataset"/"$model".py --config.add_noise_to_positions=True --config.position_noise_std=0.1 --config.dataset="$dataset"  --config.max_n_graphs=16  --config.max_ell="$l" --config.num_channels="$c" --config.target_position_predictor.num_channels="$pc"  --config.num_interactions="$i" --config.num_train_steps=10000000 --config.compute_global_embedding="$globalembed" --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/global_embed="$globalembed"  > "$expname"_model="$model"_l="$l"_pc="$pc"_c="$c"_i="$i"_ge="$globalembed".txt 2>&1  &
          done
        done
      done
    done
  done
done
