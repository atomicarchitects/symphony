workdirs=(
    "/home/ameyad/spherical-harmonic-net/workdirs/qm9_bessel_embedding_attempt6_edm_splits_iclr2024_submission/e3schnet_and_nequip/interactions=3/l=5/position_channels=2/channels=64"
    "/home/songk/workdirs/qm9_mad_nn_edm_max_targets_4"
    "/home/songk/workdirs/qm9_mad_radius_max_targets_4"
)
cuda=2
for workdir in "${workdirs[@]}"
do
    CUDA_VISIBLE_DEVICES="$cuda$" python -m analyses.generate_molecules --workdir="$workdir" --max_num_atoms=35 --num_seeds=10000 --num_seeds_per_chunk=25 \
    > "$cuda"_generate_molecules.txt 2>&1 &
    cuda=$((cuda + 1))
    CUDA_VISIBLE_DEVICES="$cuda$" python -m analyses.conditional_generation --workdir="$workdir" \
    > "$cuda"_conditional_generation.txt 2>&1 &
    cuda=$((cuda + 1))
done