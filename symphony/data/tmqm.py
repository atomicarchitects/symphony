from typing import List

import logging
import os
import zipfile
import tqdm
import urllib
import pickle
import numpy as np
import networkx as nx
from sh import gunzip
import jraph
import ase.io
from symphony import datatypes
from symphony.data.utils import clone_url, download_url, extract_zip
from symphony.models import ptable

TMQM_URL = "https://github.com/bbskjelstad/tmqm.git"
TMQMG_URL = "https://ns9999k.webs.sigma2.no/10.11582_2022.00057/nird/home/hanneskn/tmQMg/uNatQ_graphs.zip"


def load_tmqm(root_dir: str) -> List[ase.Atoms]:
    """Load the TMQM dataset."""
    mols_as_ase = []
    data_path = root_dir
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
                    xyz_lines = xyz.split("\n")
                    assert len(xyz_lines) == int(xyz_lines[0]) + 2
                    with open(os.path.join(xyzs_path, f"X{i}_{xyz_n}.xyz"), "w") as f:
                        f.write(xyz)

    for mol_file in tqdm.tqdm(os.listdir(xyzs_path)):
        mol_as_ase = ase.io.read(os.path.join(xyzs_path, mol_file), format="xyz")
        if mol_as_ase is None:
            continue
        mols_as_ase.append(mol_as_ase)

    logging.info(f"Loaded {len(mols_as_ase)} molecules.")
    return mols_as_ase


def load_tmqmg(root_dir: str) -> List[ase.Atoms]:
    """Load the TMQMg dataset."""
    pkl_path = os.path.join(root_dir, "data.pkl")
    if os.path.exists(pkl_path):
        logging.info(f"Using downloaded data: {root_dir}")
        with open(pkl_path, "rb") as f:
            mols, neighbor_lists = pickle.load(f)
            return mols, neighbor_lists
    data_path = os.path.join(root_dir, "uNatQ_graphs")
    if os.path.exists(data_path):
        logging.info(f"Using downloaded data: {data_path}")
    else:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        logging.info(f"Downloading tmQMg zip to {root_dir}...")
        zip_path = download_url(TMQMG_URL, root_dir)

        logging.info(f"Unzipping {zip_path}...")
        extract_zip(zip_path, root_dir)

    filenames = os.listdir(data_path)
    mols = []
    neighbor_lists = []
    for mol_file in tqdm.tqdm(filenames):
        gml_mol = nx.read_gml(os.path.join(data_path, mol_file))
        gml_nodes = list(gml_mol.nodes.keys())
        gml_edges = list(gml_mol.edges.keys())

        gml_positions = []
        gml_species = []
        for node in gml_nodes:
            gml_positions.append(gml_mol.nodes[node]["node_position"])
            gml_species.append(gml_mol.nodes[node]["feature_atomic_number"])
        gml_positions = np.array(gml_positions)
        gml_species = np.array(gml_species)

        gml_senders = []
        gml_receivers = []
        for s, e, _ in gml_edges:
            gml_senders.append(int(s))
            gml_receivers.append(int(e))
        gml_senders = np.asarray(gml_senders)
        gml_receivers = np.asarray(gml_receivers)

        bound1 = ptable.groups[gml_species-1] >= 2
        bound2 = ptable.groups[gml_species-1] <= 11
        heavy = gml_species[bound1 & bound2]
        assert heavy.shape[0] == 1, print(ptable.groups[gml_species-1])
        neighbors = gml_receivers[gml_species[gml_senders] == heavy[0]]

        mol_ase = ase.Atoms(
            positions=np.asarray(gml_positions),
            numbers=np.asarray(gml_species),
        )
        mols.append(mol_ase)
        neighbor_lists.append(neighbors)

    with open(pkl_path, "wb") as f:
        pickle.dump((mols, neighbor_lists), f)

    logging.info(f"Loaded {len(mols)} molecules.")
    return mols, neighbor_lists
