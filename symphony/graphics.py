from typing import List, Tuple

import py3Dmol
from rdkit import Chem


def plot_molecules_with_py3Dmol(
    molecules: List[Chem.Mol],
    num_columns: int = 5,
    window_size: Tuple[int, int] = (250, 250),
    show_atom_types: bool = True,
) -> py3Dmol.view:
    # Reshape into a grid.
    molecules = [
        molecules[i : i + num_columns] for i in range(0, len(molecules), num_columns)
    ]

    view = py3Dmol.view(
        viewergrid=(len(molecules), len(molecules[0])),
        linked=True,
        width=len(molecules[0]) * window_size[0],
        height=len(molecules) * window_size[1],
    )
    for i, row in enumerate(molecules):
        for j, mol in enumerate(row):
            view.addModel(Chem.MolToMolBlock(mol), "mol", viewer=(i, j))

            if show_atom_types:
                for atom in mol.GetAtoms():
                    position = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    view.addLabel(
                        str(atom.GetSymbol()),
                        {
                            "fontSize": 6,
                            "fontColor": "white",
                            "backgroundOpacity": 0.2,
                            "backgroundColor": "black",
                            "position": {
                                "x": position.x,
                                "y": position.y,
                                "z": position.z,
                            },
                        },
                        viewer=(i, j),
                    )

    view.setStyle(
        {"stick": {"color": "spectrum", "radius": 0.2}, "sphere": {"scale": 0.3}}
    )
    view.zoomTo()
    return view
