##########################################################
# From G-SchNet repository                               #
# https://github.com/atomistic-machine-learning/G-SchNet #
##########################################################

import argparse
import ase

from analysis import construct_pybel_mol


def get_parser():
    """Setup parser for command line arguments"""
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "mol_path",
        help="Path to generated molecule in .xyz format",
    )

    return main_parser


def check_valence(
    mol,
    valence=[0, 1, 0, 0, 0, 0, 4, 3, 2, 1],
):
    """
    Assess whether a molecule meets valence constraints or not
    (i.e. all of their atoms have the correct number of bonds).

    Args:
        mol (ase.Atoms): an ASE molecule
        valence (numpy.ndarray): list of valenc of each atom type where the index in
            the list corresponds to the type (e.g. [0, 1, 0, 0, 0, 0, 4, 3, 2, 1] for
            qm9 molecules as H=type 1 has valency of 1, C=type 6 has valency of 4,
            N=type 7 has valency of 3 etc.)
        print_file (bool, optional): set True to suppress printing of progress string
            (default: True)
        prog_str (str, optional): specify a custom progress string (default: None)

    Returns:
        valid_mol (bool): True if molecule has passed the valence check, False otherwise
        valid_atoms (int): the number of atoms in each molecule that pass the valence check
    """
    n_atoms = len(mol.numbers)
    valid_atoms = 0
    pybel_mol = construct_pybel_mol(mol)

    for atom in pybel_mol.atoms:
        if atom.OBAtom.GetExplicitValence() == valence[atom.atomicnum]:
            valid_atoms += 1
    
    return valid_atoms == n_atoms, valid_atoms


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    mol = ase.io.read(args.mol_path)

    # check valency
    valid_mol, valid_atoms = check_valence(mol)
    print(f'{mol.symbols} {"does" if valid_mol else "does not"} satisfy valence constraints')
    print(f'{valid_atoms} of {len(mol.numbers)} atoms satisfy valence constraints')
