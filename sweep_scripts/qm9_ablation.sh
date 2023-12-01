#!/bin/bash

# Set experiment name
dataset="qm9"
expname="$dataset"_ablation

# Loop over hyperparameters
for model in "e3schnet_and_nequip"
do
  for lfocus in 1 2
  do
  for l in 1 2 3 4 5
  do
    for pc in 2
    do
      for c in 64
      do
        for i in 3
	do
           CUDA_VISIBLE_DEVICES=7 python -m symphony \
           --config=configs/"$dataset"/"$model".py --config.dataset="$dataset" \
           --config.focus_and_target_species_predictor.embedder_config.max_ell="$lfocus" \
           --config.target_position_predictor.embedder_config.max_ell="$l" \
           --config.focus_and_target_species_predictor.embedder_config.num_channels="$c" \
           --config.target_position_predictor.embedder_config.num_channels="$c" \
           --config.focus_and_target_species_predictor.embedder_config.num_interactions="$i" \
           --config.target_position_predictor.embedder_config.num_interactions="$i" \
           --config.target_position_predictor.num_channels="$pc" \
           --config.num_train_steps=1000000 \
           --config.target_position_predictor.res_beta=45 \
           --config.target_position_predictor.res_alpha=89 \
           --config.position_noise_std=0.05 \
           --config.max_n_graphs=16 \
           --config.learning_rate=0.0005 \
           --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/lfocus="$lfocus"/position_channels="$pc"/channels="$c"/  > "$expname"_model="$model"_l="$l"_lfocus="$lfocus"_pc="$pc"_c="$c"_i="$i".txt 2>&1
        done
      done
    done
  done
done
done
