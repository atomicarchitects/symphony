"""Loads the model from a workdir to perform analysis."""

import os
import pickle
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import yaml

from absl import logging
from clu import checkpoint
from flax.training import train_state

import datatypes
import input_pipeline_tf
import train


ATOMIC_NUMBERS = [1, 6, 7, 8, 9]


def cast_keys_as_int(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:
    """Returns a dictionary with string keys converted to integers, wherever possible."""
    casted_dictionary = {}
    for key, val in dictionary.items():
        try:
            val = cast_keys_as_int(val)
        except AttributeError:
            pass

        try:
            key = int(key)
        except ValueError:
            pass
        finally:
            casted_dictionary[key] = val
    return casted_dictionary


def load_from_workdir(
    workdir: str,
) -> Tuple[ml_collections.ConfigDict, train_state.TrainState, Dict[Any, Any]]:
    """Loads the scaler, model and auxiliary data from the supplied workdir."""

    if not os.path.exists(workdir):
        raise FileNotFoundError(f"{workdir} does not exist.")

    # Load config.
    saved_config_path = os.path.join(workdir, "config.yml")
    if not os.path.exists(saved_config_path):
        raise FileNotFoundError(f"No saved config found at {workdir}")

    logging.info("Saved config found at %s", saved_config_path)
    with open(saved_config_path, "r") as config_file:
        config = yaml.unsafe_load(config_file)

    # Check that the config was loaded correctly.
    assert config is not None

    # Mimic what we do in train.py.
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, dataset_rng = jax.random.split(rng)

    # Obtain graphs.
    datasets = input_pipeline_tf.get_datasets(dataset_rng, config)
    train_iter = datasets["train"].as_numpy_iterator()
    init_graphs = next(train_iter)

    # Set up dummy variables to obtain the structure.
    rng, init_rng = jax.random.split(rng)
    net = train.create_model(config, config.evaluation_mode)
    params = jax.jit(net.init)(init_rng, init_graphs)
    tx = train.create_optimizer(config)
    dummy_state = train_state.TrainState.create(
        apply_fn=net.apply, params=params, tx=tx
    )

    # Load the actual values.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=5)
    data = ckpt.restore({"best_state": dummy_state, "metrics_for_best_state": None})
    best_state = jax.tree_map(jnp.asarray, data["best_state"])

    return (
        config,
        best_state,
        cast_keys_as_int(data["metrics_for_best_state"]),
    )


def to_db(generated_frag: datatypes.Fragment, modelpath, file_name):
    pass


def to_mol_dict(generated_frag: datatypes.Fragment, modelpath, file_name):
    first_index = np.asarray(jnp.concatenate([jnp.array([0]), jnp.cumsum(generated_frag.n_node)]))
    positions = np.asarray(generated_frag.nodes.positions)
    species = np.asarray(list(map(lambda z: ATOMIC_NUMBERS[z], generated_frag.nodes.species.tolist())))
    
    generated = {}  # G-SchNet seems to expect this to be a dictionary with int keys and dictionary values
    # is this supposed to be one sequence of generated fragments at a time, or can multiple mols be considered?
    for i in range(len(first_index) - 1):
        k = int(first_index[i+1] - first_index[i])
        if k not in generated:
            generated[k] = {
                "_positions": np.array([positions[first_index[i]:first_index[i+1]]]),
                "_atomic_numbers": np.array([species[first_index[i]:first_index[i+1]]])
            }
        else:
            generated[k]["_positions"] = np.append(
                generated[k]["_positions"],
                [positions[first_index[i]:first_index[i+1]]],
                0)
            generated[k]["_atomic_numbers"] = np.append(
                generated[k]["_atomic_numbers"],
                [species[first_index[i]:first_index[i+1]]],
                0)

    gen_path = os.path.join(modelpath, 'generated/')
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    # get untaken filename and store results
    file_name = os.path.join(gen_path, file_name)
    if os.path.isfile(file_name + '.mol_dict'):
        expand = 0
        while True:
            expand += 1
            new_file_name = file_name + '_' + str(expand)
            if os.path.isfile(new_file_name + '.mol_dict'):
                continue
            else:
                file_name = new_file_name
                break
    with open(file_name + '.mol_dict', 'wb') as f:
        pickle.dump(generated, f)

    return generated
