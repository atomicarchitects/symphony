mode=nn
max_targets_per_graph=4
workdir=/home/songk/workdirs/tmqm_fragments_metal_first_reduced_elementinfo/e3schnet_and_nequip/$mode/max_targets_$max_targets_per_graph

# CUDA_VISIBLE_DEVICES=0 python -m symphony.data.generation_scripts.tmqm_fragmenter \
#     --mode=nn --max_targets_per_graph=4 --nn_cutoff=3.0 \
    # --output_dir=/data/NFS/potato/songk/tmqm_fragments_heavy_first
CUDA_VISIBLE_DEVICES=0 python -m symphony \
    --config=configs/tmqm/e3schnet_and_nequip.py \
    --config.fragment_logic=$mode \
    --config.max_targets_per_graph=$max_targets_per_graph \
    --config.num_train_steps=200000 \
    --workdir=$workdir
CUDA_VISIBLE_DEVICES=0 python -m analyses.generate_molecules \
    --workdir=$workdir \
    --max_num_atoms=200 \
    --num_seeds=50 \
    --num_seeds_per_chunk=1 \
    --init=analyses/molecules/downloaded/Ni.xyz \
# CUDA_VISIBLE_DEVICES=0 python -m analyses.conditional_generation --workdir=$workdir