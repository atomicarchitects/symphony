#!/bin/bash

# Set experiment name
dataset="platonic_solids"
expname="$dataset"_by_piece_meanangular_max_targets_5_pit_200_radius

# Loop over hyperparameters
for piece in 0 1 2 3 4
do
for model in "nequip"
do
  for lfocus in 2
  do
  for l in 2
  do
    for pc in 1
    do
      for c in 32
      do
        for i in 2
        do
            CUDA_VISIBLE_DEVICES="$piece" python -m symphony \
            --config=configs/"$dataset"/"$model".py \
            --config.add_noise_to_positions=True \
            --config.position_noise_std=0.1 \
            --config.train_pieces="($piece, $((piece + 1)))" \
            --config.val_pieces="($piece, $((piece + 1)))" \
            --config.test_pieces="($piece, $((piece + 1)))" \
            --config.dataset="$dataset" \
            --config.max_n_graphs=16 \
            --config.max_targets_per_graph=5 \
            --config.loss_kwargs.target_position_inverse_temperature=200.0 \
            --config.fragment_logic="radius" \
            --config.loss_kwargs.ignore_position_loss_for_small_fragments=False \
            --config.focus_and_target_species_predictor.embedder_config.max_ell="$lfocus" \
            --config.target_position_predictor.embedder_config.max_ell="$l" \
            --config.focus_and_target_species_predictor.embedder_config.num_channels="$c" \
            --config.target_position_predictor.embedder_config.num_channels="$c" \
            --config.focus_and_target_species_predictor.embedder_config.num_interactions="$i" \
            --config.target_position_predictor.embedder_config.num_interactions="$i" \
            --config.target_position_predictor.num_channels="$pc" \
            --config.num_train_steps=10000 \
            --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/lfocus="$lfocus"/position_channels="$pc"/channels="$c"/apply_gate="$apply_gate"/square_logits="$square_logits"/piece="$piece" \
            > "$expname"_model="$model"_l="$l"_lfocus="$lfocus"_pc="$pc"_c="$c"_i="$i"_piece="$piece".txt 2>&1 &
          done
        done
      done
    done
  done
done
done
