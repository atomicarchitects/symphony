from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import rdBase
import ase
import io
import numpy as np


from symphony import datatypes


class FocusAndTargetSpeciesPredictor(hk.Module):
    """Predicts the focus and target species distribution over all nodes."""

    def __init__(
        self,
        node_embedder_fn: Callable[[], hk.Module],
        latent_size: int,
        num_layers: int,
        activation: Callable[[jnp.ndarray], jnp.ndarray],
        num_species: int,
        species_list: jnp.ndarray,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.node_embedder = node_embedder_fn()
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.activation = activation
        self.num_species = num_species
        self.species_list = species_list

    def __call__(
        self, graphs: datatypes.Fragments, inverse_temperature: float = 1.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        num_graphs = graphs.n_node.shape[0]

        # Get the node embeddings.
        node_embeddings = self.node_embedder(graphs)

        num_nodes, _ = node_embeddings.shape
        node_embeddings = node_embeddings.filter(keep="0e")
        focus_and_target_species_logits = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[self.latent_size] * (self.num_layers - 1)
            + [self.num_species],
            act=self.activation,
            output_activation=False,
        )(node_embeddings).array
        stop_logits = jnp.zeros((num_graphs,))

        assert focus_and_target_species_logits.shape == (num_nodes, self.num_species)
        assert stop_logits.shape == (num_graphs,)

        # Scale the logits by the inverse temperature.
        focus_and_target_species_logits *= inverse_temperature
        stop_logits *= inverse_temperature

        # experimental rdkit valence check
        result_shape = jax.ShapeDtypeStruct((num_nodes,), jnp.bool)
        valence_mask = jax.pure_callback(
            check_valences,
            result_shape,
            graphs.nodes.positions,
            self.species_list[graphs.nodes.species],
            graphs.n_node,
        )
        focus_and_target_species_logits = jnp.where(
            valence_mask[:, None], -1000, focus_and_target_species_logits
        )

        return focus_and_target_species_logits, stop_logits


def _ase_to_rdkit_molecule(ase_mol: ase.Atoms) -> Chem.Mol:
    """Converts a molecule from ase format to an RDKit molecule."""
    with io.StringIO() as f:
        ase.io.write(f, ase_mol, format="xyz")
        f.seek(0)
        xyz = f.read()
    mol = Chem.MolFromXYZBlock(xyz)
    return Chem.Mol(mol)

def _check_valences_single(positions, numbers):
    """for a single molecule, return true for each atom if its valence is correct & false otherwise"""
    blocker = rdBase.BlockLogs()
    ase_mol = ase.Atoms(positions = positions, numbers = numbers)
    mol = _ase_to_rdkit_molecule(ase_mol)
    bonds_determined = False
    for charge in [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5]:
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
            bonds_determined = True
            break
        except:
            continue
    if mol.GetNumBonds() == 0 or not bonds_determined: return np.zeros(len(numbers), dtype=bool)
    valences = [atom.GetExplicitValence() for atom in mol.GetAtoms()]
    full_valences = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}  # TODO hardcoded
    valence_mask = np.array(valences) >= np.array([full_valences[n] for n in numbers])
    neighbors_mask = np.array([len(atom.GetNeighbors()) >= full_valences[atom.GetAtomicNum()] for atom in mol.GetAtoms()])
    return valence_mask & neighbors_mask

def check_valences(positions, numbers, n_node):
    """for a batch of molecules, return true for each atom if its valence is correct & false otherwise"""
    index_list = np.split(np.arange(n_node.sum()), np.cumsum(n_node[:-1]))
    valences_list = [
        _check_valences_single(
            positions[index],
            numbers[index]
        ) for index in index_list
    ]
    indices_graphs = np.concatenate(
        index_list[:(n_node!=0)[:-1].sum()])
    valence_mask = np.concatenate(valences_list)
    # print("valence mask:", valence_mask[indices_graphs])
    # print("positions:", positions[indices_graphs])
    # print("numbers:", numbers[indices_graphs])
    # print("n_node:", n_node)
    return valence_mask

