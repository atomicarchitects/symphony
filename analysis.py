"""Loads the model from a workdir to perform analysis."""

import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jraph
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import yaml

from absl import logging
from clu import checkpoint
from flax.training import train_state

import input_pipeline_tf
import train


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
    workdir: str, init_graphs: Optional[jraph.GraphsTuple] = None
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
    config = ml_collections.ConfigDict(config)
    assert config is not None

    # Mimic what we do in train.py.
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, dataset_rng = jax.random.split(rng)

    # Obtain graphs.
    if init_graphs is None:
        datasets = input_pipeline_tf.get_datasets(dataset_rng, config)
        train_iter = datasets["train"].as_numpy_iterator()
        init_graphs = next(train_iter)

    # Set up dummy variables to obtain the structure.
    rng, init_rng = jax.random.split(rng)
    net = train.create_model(config, run_in_evaluation_mode=True)
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
