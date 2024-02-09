import ase
import ase.io
import sys, os
from itertools import product
import logging
import numpy as np
from openbabel import pybel
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolfiles import SDWriter, MolToXYZFile
import tqdm
from typing import List
from symphony.data.utils import extract_zip, download_url

# code extensively borrowed from https://github.com/igashov/DiffLinker and https://github.com/fimrie/DeLinker
DOWNLOAD_URL = "https://zenodo.org/api/records/7121271/files-archive"


def read_sdf(sdf_path):
    with Chem.SDMolSupplier(sdf_path, sanitize=False) as supplier:
        for molecule in supplier:
            yield molecule


def match_mol_to_frags(mol, frags, linker):
    new_indices = []
    mol_positions = mol.GetConformer().GetPositions()
    frags_positions = frags.GetConformer().GetPositions()
    linker_positions = linker.GetConformer().GetPositions()
    for atom in frags.GetAtoms():
        for new_atom in mol.GetAtoms():
            if atom.GetAtomicNum() == new_atom.GetAtomicNum() and \
                np.array_equal(mol_positions[new_atom.GetIdx()], frags_positions[atom.GetIdx()]):
                new_indices.append(new_atom.GetIdx())
                break
    for atom in linker.GetAtoms():
        for new_atom in mol.GetAtoms():
            if atom.GetAtomicNum() == new_atom.GetAtomicNum() and \
                np.array_equal(mol_positions[new_atom.GetIdx()], linker_positions[atom.GetIdx()]):
                new_indices.append(new_atom.GetIdx())
                break
    mol = Chem.rdmolops.RenumberAtoms(mol,new_indices)
    return mol, np.asarray(np.arange(frags_positions.shape[0]))


# def align_mol_to_frags(mol, smi_mol, smi_linker, smi_frags):
#     smi_frag_list = smi_frags.split('.')
#     assert len(smi_frag_list) == 2
#     # mol_from_smiles = Chem.MolFromSmiles(smi_mol)
#     # linker = Chem.MolFromSmiles(smi_linker)
#     # frag1 = Chem.MolFromSmiles(smi_frag_list[0])
#     # frag2 = Chem.MolFromSmiles(smi_frag_list[1])
#     mol_from_smiles = Chem.MolFromSmarts(pybel.readstring('smi', smi_mol).write('smi'))
#     linker = Chem.MolFromSmarts(pybel.readstring('smi', smi_linker).write('smi'))
#     frag1 = Chem.MolFromSmiles(pybel.readstring('smi', smi_frag_list[0]).write('smi'))
#     frag2 = Chem.MolFromSmiles(pybel.readstring('smi', smi_frag_list[1]).write('smi'))

#     qp = Chem.AdjustQueryParameters()
#     qp.makeDummiesQueries=True

#     # Renumber molecule based on frags (incl. dummy atoms)
#     aligned_mols = []

#     sub_idx = []
#     # Get matches to fragments and linker
#     qfrag1 = Chem.AdjustQueryProperties(frag1,qp)
#     frag1_matches = list(mol_from_smiles.GetSubstructMatches(qfrag1, uniquify=False))
#     qfrag2 = Chem.AdjustQueryProperties(frag2,qp)
#     frag2_matches = list(mol_from_smiles.GetSubstructMatches(qfrag2, uniquify=False))
#     qlinker = Chem.AdjustQueryProperties(linker,qp)
#     linker_matches = list(mol_from_smiles.GetSubstructMatches(qlinker, uniquify=False))

#     # frag_match = []
#     # linker_match = []
#     # Loop over matches
#     for frag1_match, frag2_match, linker_match in product(frag1_matches, frag2_matches, linker_matches):
#         # Check if match
#         f1_match = [idx for num, idx in enumerate(frag1_match) if frag1.GetAtomWithIdx(num).GetAtomicNum() != 0]
#         f2_match = [idx for num, idx in enumerate(frag2_match) if frag2.GetAtomWithIdx(num).GetAtomicNum() != 0]
#         l_match = [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in f_match]
#         # If perfect match, break
#         if len(set(list(f1_match)+list(f2_match)+list(l_match))) == mol.GetNumHeavyAtoms():
#             break
#     # Add frag indices
#     sub_idx += frag1_match
#     frag2_match = [idx for num, idx in enumerate(frag1_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in sub_idx]
#     sub_idx += frag2_match
#     # Add linker indices to end
#     sub_idx += [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in sub_idx]

#     aligned_mol = Chem.rdmolops.RenumberAtoms(mol, sub_idx)

#     # Renumber dummy atoms to end
#     dummy_idx = []
#     for atom in frag1.GetAtoms():
#         if atom.GetAtomicNum() == 0:
#             dummy_idx.append(atom.GetIdx())
#     # sub_idx = list(range(aligned_mols[1].GetNumHeavyAtoms()+2))
#     # for idx in dummy_idx:
#     #     sub_idx.remove(idx)
#     #     sub_idx.append(idx)
#     # mol_range = list(range(mol.GetNumHeavyAtoms()))
#     # idx_to_add = list(set(mol_range).difference(set(sub_idx)))
#     # sub_idx.extend(idx_to_add)

#     return aligned_mol, len(frag1_match)+len(frag2_match)


def prepare_data(root_dir):
    data = {'mols': [], 'frags': [], 'frag_indices': [], 'anchors': []}
    for split in ['train', 'val', 'test']:
        # 438610 train, 400 val, 400 test
        mol_path = os.path.join(root_dir, f'zinc_final_{split}_mol.sdf')
        frag_path = os.path.join(root_dir, f'zinc_final_{split}_frag.sdf')
        link_path = os.path.join(root_dir, f'zinc_final_{split}_link.sdf')
        csv_path = os.path.join(root_dir, f'zinc_final_{split}_table.csv')  # for anchor points
        data_df = pd.read_csv(csv_path)
        writer = SDWriter(f'mol_{split}.sdf')
        for mol, frags, linker, (i, row) in tqdm.tqdm(zip(
            read_sdf(mol_path),
            read_sdf(frag_path),
            read_sdf(link_path),
            data_df.iterrows()
        )):
            mol, frag_indices = match_mol_to_frags(mol, frags, linker)
            xyz_file_mol = os.path.join(root_dir, 'xyz', f'tmp_mol_{split}_{i}.xyz')
            xyz_file_frags = os.path.join(root_dir, 'xyz', f'tmp_frags_{split}_{i}.xyz')
            MolToXYZFile(mol, xyz_file_mol)
            MolToXYZFile(frags, xyz_file_frags)
            data['mols'].append(ase.io.read(xyz_file_mol))
            data['frags'].append(ase.io.read(xyz_file_frags))
            data['frag_indices'].append(frag_indices)
            data['anchors'].append((row['anchor_1'], row['anchor_2']))
        writer.close()
    return data


def load_linker(root_dir):
    '''returns ase atoms:
    {
        mols,
        frags,
        frag_indices (indices of the atoms in the molecule that correspond to the fragments),
        anchors
    }'''
    file_name = "zinc_data.pkl"
    file_path = os.path.join(root_dir, file_name)
    if os.path.exists(file_path):
        logging.info(f"Loading zinc data from {file_path}...")
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif os.path.exists(root_dir):
        logging.info(f"Using downloaded data: {root_dir}")
    else:
        os.makedirs(root_dir, exist_ok=True)
        zip_path = download_url(DOWNLOAD_URL, root_dir, progress=False, filename='linker.zip')
        logging.info(f"Unzipping {zip_path}...")
        extract_zip(zip_path, root_dir)

    logging.info(f"Preparing zinc data...")
    with open(file_path, 'wb') as f:
        pickle.dump(prepare_data(root_dir), f)
    return prepare_data(root_dir)


if __name__ == '__main__':
    load_linker('linker_data')
