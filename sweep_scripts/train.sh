cuda=4
dataset="silica"
for mode in "nn" "radius"
do
    for max_targets in 1 4
    do
        CUDA_VISIBLE_DEVICES="$cuda" python -m symphony --config=configs/"$dataset"/e3schnet_and_nequip.py --config.fragment_logic="$mode" --config.max_targets_per_graph="$max_targets" --workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/"$dataset"/e3schnet_and_nequip/"$mode"/max_targets_"$max_targets" \
        > "$cuda"_train.txt 2>&1 &
        cuda=$((cuda + 1))
    done
done