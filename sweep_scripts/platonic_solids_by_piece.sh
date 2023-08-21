#!/bin/bash

# Set experiment name
dataset="platonic_solids"
expname="$dataset"_without_extra_irreps_with_noise

# Loop over hyperparameters
for model in "nequip"
do
  for l in 4
  do
    for pc in 5
    do
      for c in 64
      do
        for i in 3
	do
          for piece in 0 1 2 3 4
          do
            CUDA_VISIBLE_DEVICES="$((piece + 2))" python -m symphony --config=configs/"$dataset"/"$model".py --config.add_noise_to_positions=True --config.position_noise_std=0.1  --config.use_pseudoscalars_and_pseudovectors=False --config.train_pieces="($piece, $((piece + 1)))" --config.val_pieces="($piece, $((piece + 1)))" --config.test_pieces="($piece, $((piece + 1)))" --config.dataset="$dataset"  --config.max_n_graphs=16  --config.max_ell="$l" --config.num_channels="$c" --config.target_position_predictor.num_channels="$pc"  --config.num_interactions="$i" --config.num_train_steps=100000 --config.compute_global_embedding=False --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/piece="$piece"  > "$expname"_model="$model"_l="$l"_pc="$pc"_c="$c"_i="$i"_piece="$piece".txt 2>&1  &
          done
        done
      done
    done
  done
done


