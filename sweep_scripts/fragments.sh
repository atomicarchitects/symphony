for mode in "nn"
do
    for max_targets in 4
    do
        # CUDA_VISIBLE_DEVICES="$cuda$" python -m symphony.data.generation_scripts.silica_fragmenter --config=configs/silica/default.py --config.fragment_logic="$mode" --mode="$mode" --max_targets_per_graph="$max_targets" \
        python -m symphony.data.generation_scripts.silica_fragmenter \
        --config=configs/silica/default.py --mode="$mode" --max_targets_per_graph="$max_targets" \
        --shuffle --nn_cutoff=3.0 --tetrahedra_only=True \
        --output_dir=/data/NFS/potato/songk/silica_shuffled_tetrahedra_nn3 \
        > "$mode"_"$max_targets"_generate_fragments.txt 2>&1 &
        # cuda=$((cuda + 1))
    done
done