import ase
import sys, os
from itertools import product
from rdkit import Chem
import tqdm

# code extensively borrowed from https://github.com/fimrie/DeLinker


def align_mol_to_frags(smi_molecule, smi_linker, smi_frags):
    try:
        # Load SMILES as molecules
        mol = Chem.MolFromSmiles(smi_molecule)
        frags = Chem.MolFromSmiles(smi_frags)
        linker = Chem.MolFromSmiles(smi_linker)
        # Include dummy atoms in query
        du = Chem.MolFromSmiles('*')
        qp = Chem.AdjustQueryParameters()
        qp.makeDummiesQueries=True
    
        # Renumber molecule based on frags (incl. dummy atoms)
        aligned_mols = []

        sub_idx = []
        # Get matches to fragments and linker
        qfrag = Chem.AdjustQueryProperties(frags,qp)
        frags_matches = list(mol.GetSubstructMatches(qfrag, uniquify=False))
        qlinker = Chem.AdjustQueryProperties(linker,qp)
        linker_matches = list(mol.GetSubstructMatches(qlinker, uniquify=False))

        # Loop over matches
        for frag_match, linker_match in product(frags_matches, linker_matches):
            # Check if match
            f_match = [idx for num, idx in enumerate(frag_match) if frags.GetAtomWithIdx(num).GetAtomicNum() != 0]
            l_match = [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in f_match]
            # If perfect match, break
            if len(set(list(f_match)+list(l_match))) == mol.GetNumHeavyAtoms():
                break
        # Add frag indices
        sub_idx += frag_match
        # Add linker indices to end
        sub_idx += [idx for num, idx in enumerate(linker_match) if linker.GetAtomWithIdx(num).GetAtomicNum() != 0 and idx not in sub_idx]

        aligned_mols.append(Chem.rdmolops.RenumberAtoms(mol, sub_idx))
        aligned_mols.append(frags)

        nodes_to_keep = [i for i in range(len(frag_match))]
        
        # Renumber dummy atoms to end
        dummy_idx = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_idx.append(atom.GetIdx())
        for i, mol in enumerate(aligned_mols):
            sub_idx = list(range(aligned_mols[1].GetNumHeavyAtoms()+2))
            for idx in dummy_idx:
                sub_idx.remove(idx)
                sub_idx.append(idx)
            if i == 0:
                mol_range = list(range(mol.GetNumHeavyAtoms()))
            else:
                mol_range = list(range(mol.GetNumHeavyAtoms()+2))
            idx_to_add = list(set(mol_range).difference(set(sub_idx)))
            sub_idx.extend(idx_to_add)
            aligned_mols[i] = Chem.rdmolops.RenumberAtoms(mol, sub_idx)

        # Get exit vectors
        exit_vectors = []
        for atom in aligned_mols[1].GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetIdx() in nodes_to_keep:
                    nodes_to_keep.remove(atom.GetIdx())
                for nei in atom.GetNeighbors():
                    exit_vectors.append(nei.GetIdx())

        if len(exit_vectors) != 2:
            print("Incorrect number of exit vectors")

        return (aligned_mols[0], aligned_mols[1]), nodes_to_keep, exit_vectors

    except:
        print("Could not align")
        return ([],[]), [], []


def preprocess(raw_data, dataset, name, test=False):
    print('Parsing smiles as graphs.')
    processed_data =[]
    total = len(raw_data)
    for i, (smi_mol, smi_frags, smi_link, abs_dist) in enumerate([(mol['smi_mol'], mol['smi_frags'], 
                                                                   mol['smi_linker'], mol['abs_dist']) for mol in raw_data]):
        if test:
            smi_mol = smi_frags
            smi_link = ''
        (mol_full, frags), nodes_to_keep, exit_points = align_mol_to_frags(smi_mol, smi_link, smi_frags)
        if mol_full == []:
            continue


def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_lines = len(lines)
    data = {'mol': [], 'linker': [], 'frags': [], 'abs_dist': [], 'angle': []}
    for line in tqdm.tqdm(lines):
        toks = line.strip().split(' ')
        if len(toks) == 3:
            smi_frags, abs_dist, angle = toks
            smi_mol = smi_frags
            smi_linker = ''
        elif len(toks) == 5:
            smi_mol, smi_linker, smi_frags, abs_dist, angle = toks
        else:
            print("Incorrect input format. Please check the README for useage.")
            exit()

        data.append({'smi_mol': smi_mol, 'smi_linker': smi_linker, 
                     'smi_frags': smi_frags,
                     'abs_dist': abs_dist, 'angle': angle})
    return data
          

def load_delinker(split, dataset):
    assert split in ['train', 'valid', 'test']
    assert dataset in ['zinc', 'casf']
    data_path = os.path.join('delinker_data', f'data_{dataset}_final_{split}.txt')
    name = f'{dataset}_{split}'

    print("Preparing: %d", name)
    raw_data = read_file(data_path)
    return raw_data
