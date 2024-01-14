cuda=0
for mode in "nn_edm" "radius"
do
    for max_targets in 1 4
    do
        CUDA_VISIBLE_DEVICES="$cuda$" python -m symphony.data.generation_scripts.qm9_fragmenter --mode="$mode" --max_targets_per_graph="$max_targets" \
        > "$cuda"_generate_fragments.txt 2>&1 &
        cuda=$((cuda + 1))
    done
done