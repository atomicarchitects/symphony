"""Loads the model from a workdir to perform analysis."""

import os

from typing import Any, Dict, Optional, Tuple, Sequence

import pickle
import jraph
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import yaml
import pandas as pd
import glob

from absl import logging
from clu import checkpoint
from flax.training import train_state

import sys

sys.path.append("..")

import datatypes
import input_pipeline_tf
import train
import models
from configs import default


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


def get_results_as_dataframe(
    model: str, metrics: Sequence[str], basedir: str
) -> Dict[str, pd.DataFrame]:
    """Returns the results for the given model as a pandas dataframe for each split."""

    def extract_hyperparameters(
        config: ml_collections.ConfigDict,
    ) -> Tuple[int, int, int]:
        """Returns the hyperparameters extracted from the config."""

        if "num_interactions" in config:
            num_interactions = config.num_interactions
        else:
            num_interactions = config.n_interactions

        max_l = config.max_ell

        if "num_channels" in config:
            num_channels = config.num_channels
        else:
            num_channels = config.n_atom_basis
            assert num_channels == config.n_filters

        return num_interactions, max_l, num_channels

    results = {"val": [], "test": []}
    for config_file_path in glob.glob(
        os.path.join(basedir, "**", model, "**", "*.yml"), recursive=True
    ):
        workdir = os.path.dirname(config_file_path)

        config, best_state, _, metrics_for_best_state = load_from_workdir(workdir)
        num_params = sum(jax.tree_leaves(jax.tree_map(jnp.size, best_state.params)))
        num_interactions, max_l, num_channels = extract_hyperparameters(config)

        for split in results:
            metrics_for_split = [
                metrics_for_best_state[split][metric] for metric in metrics
            ]
            results[split].append(
                [num_interactions, max_l, num_channels, num_params] + metrics_for_split
            )

    for split in results:
        results[split] = np.array(results[split])
        results[split] = pd.DataFrame(
            results[split],
            columns=["num_interactions", "max_l", "num_channels", "num_params"]
            + metrics,
        )
        results[split] = results[split].astype(
            {
                "num_interactions": int,
                "max_l": int,
                "num_channels": int,
                "num_params": int,
                **{metric: float for metric in metrics},
            }
        )

    return results


def load_metrics_from_workdir(
    workdir: str,
) -> Tuple[
    ml_collections.ConfigDict,
    train_state.TrainState,
    train_state.TrainState,
    Dict[Any, Any],
]:
    """Loads only the config and the metrics for the best model."""

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
    config = ml_collections.ConfigDict(config)
    config.root_dir = default.get_root_dir()

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=5)
    data = ckpt.restore({"metrics_for_best_state": None})

    return config, cast_keys_as_int(data["metrics_for_best_state"])


def load_from_workdir(
    workdir: str,
    load_pickled_params: bool = True,
    init_graphs: Optional[jraph.GraphsTuple] = None,
) -> Tuple[
    ml_collections.ConfigDict,
    train_state.TrainState,
    train_state.TrainState,
    Dict[Any, Any],
]:
    """Loads the config, best model (in train mode), best model (in eval mode) and metrics for the best model."""

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
    config = ml_collections.ConfigDict(config)
    config.root_dir = default.get_root_dir()

    # Mimic what we do in train.py.
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, dataset_rng = jax.random.split(rng)

    # Set up dummy variables to obtain the structure.
    rng, init_rng = jax.random.split(rng)

    net = train.create_model(config, run_in_evaluation_mode=False)
    eval_net = train.create_model(config, run_in_evaluation_mode=True)

    # If we have pickled parameters already, we don't need init_graphs to initialize the model.
    # Note that we restore the model parameters from the checkpoint anyways.
    # We only use the pickled parameters to initialize the model, so only the keys of the pickled parameters are important.
    if load_pickled_params:
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        pickled_params_file = os.path.join(checkpoint_dir, "params.pkl")
        if not os.path.exists(pickled_params_file):
            raise FileNotFoundError(f"No pickled params found at {pickled_params_file}")

        logging.info(
            "Initializing dummy model with pickled params found at %s",
            pickled_params_file,
        )

        with open(pickled_params_file, "rb") as f:
            params = jax.tree_map(np.array, pickle.load(f))
    else:
        if init_graphs is None:
            logging.info("Initializing dummy model with init_graphs from dataloader")
            datasets = input_pipeline_tf.get_datasets(dataset_rng, config)
            train_iter = datasets["train"].as_numpy_iterator()
            init_graphs = next(train_iter)
        else:
            logging.info("Initializing dummy model with provided init_graphs")

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
    best_state_in_eval_mode = best_state.replace(apply_fn=eval_net.apply)

    return (
        config,
        best_state,
        best_state_in_eval_mode,
        cast_keys_as_int(data["metrics_for_best_state"]),
    )


def to_db(generated_frag: datatypes.Fragment, model_path: str, file_name: str):
    raise NotImplementedError("to_db() is not implemented yet.")


def to_mol_dict(generated_frag: datatypes.Fragment, model_path: str, file_name: str):
    first_index = np.asarray(
        jnp.concatenate([jnp.array([0]), jnp.cumsum(generated_frag.n_node)])
    )
    positions = np.asarray(generated_frag.nodes.positions)
    species = np.asarray(
        list(
            map(
                lambda z: models.ATOMIC_NUMBERS[z],
                generated_frag.nodes.species.tolist(),
            )
        )
    )

    generated = (
        {}
    )  # G-SchNet seems to expect this to be a dictionary with int keys and dictionary values
    # is this supposed to be one sequence of generated fragments at a time, or can multiple mols be considered?
    for i in range(len(first_index) - 1):
        k = int(first_index[i + 1] - first_index[i])
        if k not in generated:
            generated[k] = {
                "_positions": np.array(
                    [positions[first_index[i] : first_index[i + 1]]]
                ),
                "_atomic_numbers": np.array(
                    [species[first_index[i] : first_index[i + 1]]]
                ),
            }
        else:
            generated[k]["_positions"] = np.append(
                generated[k]["_positions"],
                [positions[first_index[i] : first_index[i + 1]]],
                0,
            )
            generated[k]["_atomic_numbers"] = np.append(
                generated[k]["_atomic_numbers"],
                [species[first_index[i] : first_index[i + 1]]],
                0,
            )

    gen_path = os.path.join(model_path, "generated/")
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    # get untaken filename and store results
    file_name = os.path.join(gen_path, file_name)
    if os.path.isfile(file_name + ".mol_dict"):
        expand = 0
        while True:
            expand += 1
            new_file_name = file_name + "_" + str(expand)
            if os.path.isfile(new_file_name + ".mol_dict"):
                continue
            else:
                file_name = new_file_name
                break
    with open(file_name + ".mol_dict", "wb") as f:
        pickle.dump(generated, f)

    return generated
