"""Root directories for datasets."""

from typing import Optional
import os


def get_root_dir(dataset: str) -> Optional[str]:
    """Get the root directory for the dataset."""
    hostname, username = os.uname()[1], os.environ.get("USER")

    if hostname == "radish.mit.edu":
        return f"/data/NFS/radish/symphony/root_dirs/{dataset}"
    if hostname == "potato.mit.edu":
        if "qm9" in dataset: dataset = "qm9"
        return f"/radish/symphony/root_dirs/{dataset}"
    if hostname == "eofe10.mit.edu":
        return f"/pool001/songk/root_dirs/{dataset}"
    if hostname[-23:] == "delta.ncsa.illinois.edu":
        return f"/projects/bbyc/symphony/root_dirs/{dataset}"
    if hostname == "eofe10.mit.edu":
        return f"/pool001/songk/root_dirs/{dataset}"
    if username == "ameyad":
        return f"/Users/ameyad/Documents/spherical-harmonic-net/root_dirs/{dataset}"
    if username == "songk":
        return "/Users/songk/atomicarchitects/spherical_harmonic_net/root_dirs/{dataset}"
    
    return None


def get_root_dir_tf(dataset: str, fragment_logic: str) -> Optional[str]:
    """Get the root directory for the TF datasets."""
    hostname, username = os.uname()[1], os.environ.get("USER")

    if dataset == "qm9":
        if hostname == "radish":
            return f"/data/NFS/radish/qm9_fragments/{fragment_logic}"
        if hostname == "potato.mit.edu":
            return f"/data/NFS/potato/songk/qm9_fragments_multifocus_mini/{fragment_logic}"
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
        if hostname == "radish":
            return f"/home/ameyad/spherical-harmonic-net/temp/platonic_solids/{fragment_logic}"
        if username == "ameyad":
            return f"/Users/ameyad/Documents/spherical-harmonic-net/temp/platonic_solids/{fragment_logic}"
    if dataset == "tmqm":
        if hostname == "potato.mit.edu":
            return f"/data/NFS/potato/songk/tmqm_fragments_multifocus/{fragment_logic}"
        return f"/pool001/songk/tmqmg_fragments_heavy_first/{fragment_logic}"
    return None
