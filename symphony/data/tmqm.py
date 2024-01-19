from typing import List

import logging
import os
import zipfile
import urllib
import numpy as np
from sh import gunzip
import ase.io
from symphony.data.utils import clone_url, molecule_sanity

TMQM_URL = "https://github.com/bbskjelstad/tmqm.git"

def load_tmqm(
    root_dir: str
) -> List[ase.Atoms]:
    """Load the TMQM dataset."""
    mols_as_ase = []
    data_path = os.path.join(root_dir, "tmqm", "data")
    xyzs_path = os.path.join(data_path, "xyz")
    if os.path.exists(xyzs_path):
        logging.info(f"Using downloaded data: {xyzs_path}")
    else:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        logging.info(f"Cloning TMQM repository to {root_dir}...")
        _ = clone_url(TMQM_URL, root_dir)
        if not os.path.exists(xyzs_path):
            os.makedirs(xyzs_path)

        for i in range(1, 3):
            gz_path = os.path.join(data_path, f"tmQM_X{i}.xyz.gz")
            logging.info(f"Unzipping {gz_path}...")
            gunzip(gz_path)

            mol_file = os.path.join(data_path, f"tmQM_X{i}.xyz")
            with open(mol_file, "r") as f:
                all_xyzs = f.read().split("\n\n")
                for xyz_n, xyz in enumerate(all_xyzs):
                    if xyz == "":
                        continue
                    xyz_lines = xyz.split('\n')
                    assert len(xyz_lines) == int(xyz_lines[0]) + 2
                    with open(os.path.join(xyzs_path, f"X{i}_{xyz_n}.xyz"), "w") as f:
                        f.write(xyz)

    import tqdm
    for mol_file in tqdm.tqdm(os.listdir(xyzs_path)):
        mol_as_ase = ase.io.read(os.path.join(xyzs_path, mol_file), format="xyz")
        if mol_as_ase is None:
            continue
        mols_as_ase.append(mol_as_ase)

    logging.info(f"Loaded {len(mols_as_ase)} molecules.")
    return mols_as_ase