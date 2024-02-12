cuda=1
symm="Cubic"
for mode in "nn" # "radius"
do
    for max_targets in 4
    do
        CUDA_VISIBLE_DEVICES="$cuda" python -m symphony \
            --config=configs/silica/e3schnet_and_nequip.py --config.fragment_logic="$mode" \
            --config.max_targets_per_graph="$max_targets" \
            --config.train_molecules="(0, 30)" \
            --config.val_molecules="(30, 35)" \
            --config.test_molecules="(35, 40)" \
            --config.num_train_steps=200000 \
            --workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/silica_shuffled_$symm/e3schnet_and_nequip/"$mode"/max_targets_"$max_targets" \
        > "$mode"_"$max_targets"_train.txt 2>&1 &
    done
    # cuda=$((cuda + 1))
done