#!/bin/bash

# Set experiment name
dataset="qm9"
expname="$dataset"_7SEP_attempt2

# Loop over hyperparameters
for model in "e3schnet_and_nequip"
do
  for l in 5
  do
    for pc in 2
    do
      for c in 32
      do
        for i in 3
	do
           CUDA_VISIBLE_DEVICES=7 python -m symphony \
           --config=configs/"$dataset"/"$model".py --config.dataset="$dataset" \
           --config.focus_and_target_species_predictor.embedder_config.max_ell=2 \
           --config.target_position_predictor.embedder_config.max_ell="$l" \
           --config.focus_and_target_species_predictor.embedder_config.num_channels="$c" \
           --config.target_position_predictor.embedder_config.num_channels="$c" \
           --config.focus_and_target_species_predictor.embedder_config.num_interactions="$i" \
           --config.target_position_predictor.embedder_config.num_interactions="$i" \
           --config.target_position_predictor.num_channels="$pc" \
           --config.num_train_steps=1000000 \
           --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/  > "$expname"_model="$model"_l="$l"_pc="$pc"_c="$c"_i="$i".txt 2>&1  &
        done
      done
    done
  done
done
