import argparse
import pickle
import ase
from check_valence import check_valence


def get_parser():
    """Setup parser for command line arguments"""
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument(
        "qm9_path",
        help="Path to QM9 database",
    )
    return main_parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    qm9_total = 0
    qm9_valid = 0
    qm9_invalid_list = []
    qm9_invalid_indices = []

    with ase.db.connect(args.qm9_path) as conn:
        # check valence
        for row in conn.select():
            mol = row.toatoms()
            valid_mol, valid_atoms = check_valence(mol)
            qm9_total += 1
            if valid_mol:
                qm9_valid += 1
            else:
                i = row.id
                qm9_invalid_list.append((i, row))
                qm9_invalid_indices.append(i)
    print(f"{qm9_valid} of {qm9_total} molecules in QM9 satisfy valence constraints")
    with open("qm9_invalid.pkl", "wb") as f:
        pickle.dump(qm9_invalid_list, f)
    with open("qm9_invalid_indices.pkl", "wb") as f:
        pickle.dump(qm9_invalid_indices, f)
