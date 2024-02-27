workdirs=(
#     "/home/ameyad/spherical-harmonic-net/workdirs/qm9_bessel_embedding_attempt6_edm_splits_iclr2024_submission/e3schnet_and_nequip/interactions=3/l=5/position_channels=2/channels=64"
    # "/home/songk/workdirs/silica_shuffled_matscipy_jan25/e3schnet_and_nequip/nn/max_targets_4"
    "/home/songk/workdirs/silica_shuffled_Cubic/e3schnet_and_nequip/nn/max_targets_4"
)
cuda=1
for workdir in "${workdirs[@]}"
do
    # CUDA_VISIBLE_DEVICES="$cuda$" python -m analyses.generate_molecules --workdir="$workdir" --max_num_atoms=35 --num_seeds=10000 --num_seeds_per_chunk=25 \
    # > "$cuda"_generate_molecules.txt 2>&1 &
    # cuda=$((cuda + 1))
    CUDA_VISIBLE_DEVICES=$cuda python -m analyses.conditional_generation \
        --workdir=$workdir --store_intermediates --input_dir=/data/NFS/potato/songk/silica_shuffled_tetrahedra_Cubic \
        --mode=nn --max_targets_per_graph=4 --num_mols=5 \
        > "$cuda"_conditional_generation.txt 2>&1 &
    # cuda=$((cuda + 1))
done


# python -m analyses.conditional_generation --workdir=/home/songk/workdirs/silica/e3schnet_and_nequip/radius/max_targets_4
# python -m analyses.conditional_generation --workdir=/home/songk/workdirs/silica/e3schnet_and_nequip/radius/max_targets_1