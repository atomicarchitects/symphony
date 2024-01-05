#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python -m symphony --config=configs/$2/$3.py --workdir=/data/NFS/potato/songk/spherical-harmonic-net/workdirs/$4
