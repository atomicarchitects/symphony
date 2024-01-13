cuda=0
for mode in "nn_edm" "radius"
do
    for max_targets in 1 4
    do
        CUDA_VISIBLE_DEVICES="$cuda" python -m symphony --config=configs/qm9/e3schnet_and_nequip.py --config.fragment_logic="$mode" --config.max_targets_per_graph="$max_targets" --workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/qm9/e3schnet_and_nequip/"$mode"/max_targets_"$max_targets" \
        > "$cuda"_train.txt 2>&1 &
        cuda=$((cuda + 1))
    done
done