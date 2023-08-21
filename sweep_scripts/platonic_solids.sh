#!/bin/bash

# Set experiment name
dataset="platonic_solids"
expname="$dataset"_with_extra_irreps_with_noise

# Loop over hyperparameters
for model in "nequip"
do
  for l in 1 2 3 4
  do
    for pc in 5
    do
      for c in 64
      do
        for i in 3
        do
            CUDA_VISIBLE_DEVICES="$((l+1))" python -m symphony --config=configs/"$dataset"/"$model".py --config.use_pseudoscalars_and_pseudovectors=True --config.add_noise_to_positions=True --config.position_noise_std=0.1 --config.dataset="$dataset"  --config.max_n_graphs=16  --config.max_ell="$l" --config.num_channels="$c" --config.target_position_predictor.num_channels="$pc"  --config.num_interactions="$i" --config.num_train_steps=100000 --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/  > "$expname"_model="$model"_l="$l"_pc="$pc"_c="$c".txt 2>&1  &
        done
      done
    done
  done
done


