#!/bin/bash

# Load anaconda module
module load anaconda/2023a

echo "My file name prefix: " $1
echo "My seed: " $2
echo "My start: " $3
echo "My end: " $4
echo "My file name: " $5

echo "Number of threads: " $OMP_NUM_THREADS

python -u fragmenter.py --seed=$2 --start=$3 --end=$4 --output=$5
