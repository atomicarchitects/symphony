cuda=0
for mode in "nn" "radius"
do
    for max_targets in 1 4
    do
        # CUDA_VISIBLE_DEVICES="$cuda$" python -m symphony.data.generation_scripts.silica_fragmenter --config=configs/silica/default.py --config.fragment_logic="$mode" --mode="$mode" --max_targets_per_graph="$max_targets" \
        python -m symphony.data.generation_scripts.silica_fragmenter --config=configs/silica/default.py --config.fragment_logic="$mode" --mode="$mode" --max_targets_per_graph="$max_targets" \
        > "$cuda"_generate_fragments.txt 2>&1 &
        cuda=$((cuda + 1))
    done
done