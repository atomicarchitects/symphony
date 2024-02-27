cuda=1
symm="Cubic"
mode="nn"
max_targets=4
workdir="/data/NFS/potato/songk/spherical-harmonic-net/workdirs/silica_shuffled_$symm/e3schnet_and_nequip/$mode/max_targets_$max_targets"
    # CUDA_VISIBLE_DEVICES="$cuda" python -m symphony \
    #     --config=configs/silica/e3schnet_and_nequip.py --config.fragment_logic="$mode" \
    #     --config.max_targets_per_graph="$max_targets" \
    #     --config.train_molecules="(0, 30)" \
    #     --config.val_molecules="(30, 35)" \
    #     --config.test_molecules="(35, 40)" \
    #     --config.num_train_steps=200000 \
    #    --config.focus_and_target_species_predictor.embedder_config.max_ell=2 \
    #    --config.target_position_predictor.embedder_config.max_ell=2 \
    #    --config.focus_and_target_species_predictor.embedder_config.num_interactions=3 \
    #    --config.target_position_predictor.embedder_config.num_interactions=3 \
    #     --workdir=$workdir \
    # > "$mode"_"$max_targets"_train.txt 2>&1 &
    CUDA_VISIBLE_DEVICES=$cuda python -m analyses.conditional_generation \
    --workdir=$workdir --store_intermediates --input_dir=/data/NFS/potato/songk/silica_shuffled_tetrahedra_Cubic \
    --mode=nn --max_targets_per_graph=4 --num_mols=5 \
    > "$cuda"_conditional_generation.txt 2>&1 &