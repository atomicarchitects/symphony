mode='nn'
max_targets=4
workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/silica_shuffled_matscipy_jan25/e3schnet_and_nequip/$mode/max_targets_$max_targets
# CUDA_VISIBLE_DEVICES=2 python -m symphony.data.generation_scripts.silica_fragmenter \
#     --config=configs/silica/default.py --mode=$mode --max_targets_per_graph=$max_targets \
#     --output_dir=/data/NFS/potato/songk/silica_shuffled_matscipy_nn3 --shuffle --nn_cutoff=3.0
CUDA_VISIBLE_DEVICES=3 python -m symphony \
    --config=configs/silica/e3schnet_and_nequip.py \
    --config.fragment_logic=$mode \
    --config.max_targets_per_graph=$max_targets \
    --workdir=$workdir \
    --config.num_train_steps=1000000
CUDA_VISIBLE_DEVICES=3 python -m analyses.conditional_generation \
    --workdir=$workdir --store_intermediates --input_dir=/data/NFS/potato/songk/silica_shuffled_tetrahedra_nn3 \
    --mode=$mode --max_targets_per_graph=$max_targets
