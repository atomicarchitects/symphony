"""Root directories for datasets."""

from typing import Optional
import os


def get_root_dir(dataset: str) -> Optional[str]:
    """Get the root directory for the dataset."""
    hostname, username = os.uname()[1], os.environ.get("USER")

    if hostname == "radish.mit.edu":
        return f"/data/NFS/radish/symphony/root_dirs/{dataset}"
    if hostname == "potato.mit.edu":
        return f"/radish/symphony/root_dirs/{dataset}"
    if username == "ameyad":
        return f"/Users/ameyad/Documents/spherical-harmonic-net/root_dirs/{dataset}"
    if username == "songk":
        return "/Users/songk/atomicarchitects/spherical_harmonic_net/root_dirs/{dataset}"
    return None
