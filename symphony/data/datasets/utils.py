import ase
import matscipy.neighbours
import numpy as np
import ml_collections

from symphony.data.datasets import dataset, platonic_solids, qm9
from symphony import datatypes


def infer_edges_from_positions(structure: datatypes.Structures, nn_cutoff: float) -> datatypes.Structures:
    """Infer edges from node positions, using a radial cutoff."""
    assert structure.n_node.shape[0] == 1, "Only one structure is supported."

    receivers, senders = matscipy.neighbours.neighbour_list(
        quantities="ij", positions=structure.nodes.positions, cutoff=nn_cutoff, cell=np.eye(3)
    )

    return structure._replace(
        edges=np.ones(len(senders)),
        senders=np.asarray(senders),
        receivers=np.asarray(receivers),
        n_edge=np.array([len(senders)]),
    )


def ase_atoms_to_structure(
    atoms: ase.Atoms, atomic_numbers: np.ndarray, nn_cutoff: float
) -> datatypes.Structures:
    # Create edges.
    receivers, senders = matscipy.neighbours.neighbour_list(
        quantities="ij", positions=atoms.positions, cutoff=nn_cutoff, cell=np.eye(3)
    )

    # Get the species indices.
    species = np.searchsorted(atomic_numbers, atoms.numbers)

    return datatypes.Structures(
        nodes=datatypes.NodesInfo(np.asarray(atoms.positions), np.asarray(species)),
        edges=np.ones(len(senders)),
        globals=None,
        senders=np.asarray(senders),
        receivers=np.asarray(receivers),
        n_node=np.array([len(atoms)]),
        n_edge=np.array([len(senders)]),
    )


def get_dataset(config: ml_collections.ConfigDict) -> dataset.InMemoryDataset:
    """Creates the dataset of structures, as specified in the config."""

    if config.dataset == "qm9":
        return qm9.QM9Dataset()
    
    if config.dataset == "platonic_solids":
        return platonic_solids.PlatonicSolidsDataset(
            train_solids=config.train_solids,
            val_solids=config.val_solids,
            test_solids=config.test_solids,
            nn_cutoff=config.nn_cutoff,
        )

    raise ValueError(f"Unknown dataset: {config.dataset}")
