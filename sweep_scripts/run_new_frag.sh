# CUDA_VISIBLE_DEVICES=2 python -m symphony.data.generation_scripts.qm9_fragmenter \
#     --mode=nn_edm --max_targets_per_graph=4
CUDA_VISIBLE_DEVICES=1 python -m symphony \
    --config=configs/qm9/e3schnet_and_nequip.py \
    --config.fragment_logic=nn_edm \
    --config.max_targets_per_graph=4 \
    --config.num_train_steps=1000000 \
    --workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/qm9_new_frags_jan19/e3schnet_and_nequip/nn_edm/max_targets_4
CUDA_VISIBLE_DEVICES=1 python -m analyses.generate_molecules \
    --workdir=/home/songk/workdirs/qm9_new_frags_jan19/e3schnet_and_nequip/nn_edm/max_targets_4 \
    --max_num_atoms=35 \
    --num_seeds=10000 \
    --num_seeds_per_chunk=25 > qm9_new_frags 2>&1 &