#!/bin/bash

# Set experiment name
dataset="tmqm"
expname="$dataset"_sweep

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
            script=slurm_scripts/"$expname"_model="$model"_l="$l"_pc="$pc"_c="$c"_i="$i".sh
            echo "#!/bin/sh
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p sched_mit_hill
#SBATCH -o slurm_logs/"$expname"_model="$model"_l="$l"_pc="$pc"_c="$c"_i="$i".txt

cd /home/songk
source /home/songk/.bashrc
conda activate tmqm-dev
cd symphony-tmqm

python -m symphony.run_coordination \
    --config=configs/"$dataset"/"$model".py \
    --config.coord_predictor.embedder_config.max_ell="$lfocus" \
    --config.coord_predictor.embedder_config.num_channels="$c" \
    --config.coord_predictor.embedder_config.num_interactions="$i" \
    --config.num_train_steps=30000 \
    --config.position_noise_std=0.05 \
    --config.max_n_graphs=16 \
    --config.learning_rate=0.005 \
    --workdir=/pool001/songk/workdirs/"$expname"/"$model"/interactions="$i"/l="$l"/position_channels="$pc"/channels="$c"/" \
                > $script
            sbatch $script
        done
      done
    done
  done
done
done
