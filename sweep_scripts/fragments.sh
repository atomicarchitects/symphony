for mode in "nn"
do
    for max_targets in 4
    do
        for system in "Cubic" # "Triclinic" "Monoclinic" "Orthorhombic" "Tetragonal" "Trigonal"
        do
            # CUDA_VISIBLE_DEVICES="$cuda$" python -m symphony.data.generation_scripts.silica_fragmenter --config=configs/silica/default.py --config.fragment_logic="$mode" --mode="$mode" --max_targets_per_graph="$max_targets" \
            CUDA_VISIBLE_DEVICES=2 python -m symphony.data.generation_scripts.silica_fragmenter \
            --config=configs/silica/default.py --mode="$mode" --max_targets_per_graph="$max_targets" \
            --shuffle --nn_cutoff=3.0 --crystal_system=$system \
            --output_dir=/data/NFS/potato/songk/silica_shuffled_$system \
            --chunk=5 \
            > "$system"_"$mode"_"$max_targets"_generate_fragments.txt 2>&1 &
            # cuda=$((cuda + 1))
        done
    done
done