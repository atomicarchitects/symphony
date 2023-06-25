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
            if username == "songk":
                return "/home/songk/spherical-harmonic-net/qm9_data_tf/data_tf2"
            return f"/radish/qm9_fragments/{fragment_logic}"
        if username == "ameyad":
            return f"/Users/ameyad/Documents/qm9_data_tf/{fragment_logic}"
        if username == "songk":
            return "/Users/songk/atomicarchitects/spherical_harmonic_net/qm9_data_tf/data_tf2"
    if dataset == "tetris":
        if hostname == "potato.mit.edu":
            return f"/radish/tetris/{fragment_logic}"
        if username == "ameyad":
            return f"/Users/ameyad/Documents/spherical-harmonic-net/tetris/{fragment_logic}"
    if dataset == "platonic_solids":
        if hostname == "potato.mit.edu":
            return f"/radish/platonic_solids/{fragment_logic}"
        if username == "ameyad":
            return f"/Users/ameyad/Documents/spherical-harmonic-net/platonic_solids/{fragment_logic}"
    return None
