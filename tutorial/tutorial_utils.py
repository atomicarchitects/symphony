import os
import pickle
from typing import Tuple
import yaml
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import ml_collections

from symphony import models

from symphony import datatypes
import ase
import matscipy.neighbours
import numpy as np
import jraph


def ase_atoms_to_jraph_graph(
    atoms: ase.Atoms, atomic_numbers: jnp.ndarray, nn_cutoff: float
) -> jraph.GraphsTuple:
    # Create edges
    receivers, senders = matscipy.neighbours.neighbour_list(
        quantities="ij", positions=atoms.positions, cutoff=nn_cutoff, cell=np.eye(3)
    )

    # Get the species indices
    species = np.searchsorted(atomic_numbers, atoms.numbers)

    return jraph.GraphsTuple(
        nodes=datatypes.NodesInfo(np.asarray(atoms.positions), np.asarray(species)),
        edges=np.ones(len(senders)),
        globals=None,
        senders=np.asarray(senders),
        receivers=np.asarray(receivers),
        n_node=np.array([len(atoms)]),
        n_edge=np.array([len(senders)]),
    )


def load_model_at_step(
    workdir: str, step: str, run_in_evaluation_mode: bool,
) -> Tuple[hk.Transformed, optax.Params, ml_collections.ConfigDict]:
    """Loads the model at a given step.

    This is a lightweight version of load_from_workdir, that only constructs the model and not the training state.
    """
    params_file = os.path.join(workdir, f"checkpoints/params_{step}.pkl")
    with open(params_file, "rb") as f:
        params = pickle.load(f)

    with open(workdir + "/config.yml", "rt") as config_file:
        config = yaml.unsafe_load(config_file)
    assert config is not None
    config = ml_collections.ConfigDict(config)

    model = models.create_model(config, run_in_evaluation_mode=run_in_evaluation_mode)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    return model, params, config
