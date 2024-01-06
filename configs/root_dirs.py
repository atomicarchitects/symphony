"""Root directories for datasets."""

from typing import Optional
import os


def get_root_dir(dataset: str, fragment_logic: str) -> Optional[str]:
    """Get the root directory for the dataset."""
    hostname, username = os.uname()[1], os.environ.get("USER")

    if dataset == "qm9":
        if hostname == "radish.mit.edu":
            return f"/data/NFS/radish/qm9_fragments/{fragment_logic}"
        if hostname == "potato.mit.edu":
            return f"/radish/qm9_fragments_fixed/{fragment_logic}"
        if username == "ameyad":
            return f"/Users/ameyad/Documents/spherical-harmonic-net/qm9_fragments_fixed/{fragment_logic}"
        if username == "songk":
            return "/Users/songk/atomicarchitects/spherical_harmonic_net/qm9_data_tf/data_tf2"
    if dataset == "tetris":
        if hostname == "potato.mit.edu":
            return f"/radish/tetris/{fragment_logic}"
        if username == "ameyad":
            return f"/Users/ameyad/Documents/spherical-harmonic-net/temp/tetris/{fragment_logic}"
    if dataset == "platonic_solids":
        if hostname == "potato.mit.edu":
            return f"/radish/platonic_solids/{fragment_logic}"
        if username == "ameyad":
            return f"/Users/ameyad/Documents/spherical-harmonic-net/temp/platonic_solids/{fragment_logic}"
    if dataset == "silica":
        if hostname == "potato.mit.edu":
            return "/data/NFS/potato/songk/silica_fragments"
            # return "/data/NFS/potato/songk/silica_fragments_heavy_first"
    if dataset == "silica_mini":
        if hostname == "potato.mit.edu":
            return "/home/songk/spherical-harmonic-net/silica_fragments_mini"
    return None
