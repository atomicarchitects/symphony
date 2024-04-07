# #!/bin/bash
# #SBATCH -N 1
# #SBATCH -n 8
# #SBATCH -p sched_mit_hill

# cd /home/songk

# source .bashrc
# conda activate tmqm-dev
# cd symphony-tmqm

mode=nn
max_targets_per_graph=4
cuda=1

python -m symphony.data.generation_scripts.tmqm_fragmenter \
   --mode=nn --max_targets_per_graph=4 --nn_cutoff=3.5 --num_nodes_for_multifocus=4 \
   --output_dir=/data/NFS/potato/songk/tmqm_fragments_multifocus --end_seed=1 \
   --start_index=0 --end_index=10 --chunk=10
   # --start_index=0 --end_index=1000

# CUDA_VISIBLE_DEVICES=0 python -m symphony.data.generation_scripts.tmqm_ni_fragmenter \
#    --mode=nn --max_targets_per_graph=4 --nn_cutoff=3.5 \
#    --output_dir=/data/NFS/potato/songk/tmqm_ni_fragments_single --end_seed=8 \
#     --end_index=1 --chunk=1