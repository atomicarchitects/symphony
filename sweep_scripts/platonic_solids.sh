#!/bin/bash

# Set experiment name
dataset="platonic_solids"
expname="$dataset"_pc_ablation

# Loop over hyperparameters
for model in "e3schnet_and_nequip"
do
  for lfocus in 2
  do
  for l in 1 2 3 4 5
  do
    for pc in 1 2 3 4 5
    do
      for c in 64
      do
        for i in 3
	do
           CUDA_VISIBLE_DEVICES=4 python -m symphony \
           --config=configs/"$dataset"/"$model".py --config.dataset="$dataset" \
           --config.focus_and_target_species_predictor.embedder_config.max_ell="$lfocus" \
           --config.target_position_predictor.embedder_config.max_ell="$l" \
           --config.focus_and_target_species_predictor.embedder_config.num_channels="$c" \
           --config.target_position_predictor.embedder_config.num_channels="$c" \
           --config.focus_and_target_species_predictor.embedder_config.num_interactions="$i" \
           --config.target_position_predictor.embedder_config.num_interactions="$i" \
           --config.target_position_predictor.num_channels="$pc" \
           --config.num_train_steps=10000 \
           --config.position_noise_std=0.05 \
           --config.max_n_graphs=16 \
           --config.learning_rate=0.005 \
           --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/  > "$expname"_model="$model"_l="$l"_pc="$pc"_c="$c"_i="$i".txt 2>&1
        done
      done
    done
  done
done
done

