from typing import List

import logging
import os
import pandas as pd
from sh import gunzip
import ase.io
from symphony.data.utils import download_url, extract_tar

QCD_URL = "https://muellergroup.jhu.edu/qcd/data_csv.tar.gz"


def load_qcd(
    root_dir: str
) -> List[ase.Atoms]:
    """Load the QCD dataset."""
    mols_as_ase = []
    qcd_dir = os.path.join(root_dir, "qcd")
    data_path = os.path.join(qcd_dir, "xyz")
    if os.path.exists(data_path):
        logging.info(f"Using downloaded data: {data_path}")
    else:
        if not os.path.exists(qcd_dir):
            os.makedirs(qcd_dir)
        gz_path = download_url(QCD_URL, qcd_dir)
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        logging.info(f"Unzipping {gz_path}...")
        gunzip(gz_path)
        extract_tar(gz_path[:-3], qcd_dir)

        mol_csv = pd.read_csv(os.path.join(gz_path[:-7], 'data.csv'))
        for i, row in mol_csv.iterrows():
            xyz = row['structure_xyz'].split(';')
            if xyz[-1] == '':
                xyz = xyz[:-1]
            with open(os.path.join(data_path, f"cluster_{i}.xyz"), "w") as f:
                f.write("\n".join(xyz))

    import tqdm
    for mol_file in tqdm.tqdm(os.listdir(data_path)):
        mol_as_ase = ase.io.read(os.path.join(data_path, mol_file), format="xyz")
        if mol_as_ase is None:
            continue
        mols_as_ase.append(mol_as_ase)

    logging.info(f"Loaded {len(mols_as_ase)} molecules.")
    return mols_as_ase
