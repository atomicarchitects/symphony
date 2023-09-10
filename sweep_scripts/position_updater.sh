#!/bin/bash

# Set experiment name
dataset="qm9"
expname="$dataset"_9SEP_position_denoiser

# Loop over hyperparameters
for model in "position_updater"
do
  for l in 5
  do
    for pc in 2
    do
      for c in 32
      do
        for i in 3
	do
           CUDA_VISIBLE_DEVICES=1 python -m symphony  \
           --config=configs/"$dataset"/"$model".py --config.dataset="$dataset" \
           --config.num_train_steps=10000000 \
           --workdir=workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/  > "$expname"_model="$model"_l="$l"_pc="$pc"_c="$c"_i="$i".txt 2>&1  &
        done
      done
    done
  done
done
