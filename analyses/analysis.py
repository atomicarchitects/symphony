"""A bunch of analysis scripts."""

import glob
import os
import pickle
import sys
from typing import Any, Dict, Optional, Sequence, Tuple
import os

import haiku as hk
import ase
import ase.build
import optax
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import pandas as pd
import yaml
from absl import logging
from clu import checkpoint
from flax.training import train_state

sys.path.append("..")

from symphony.data import qm9
from symphony import models
from symphony import train
from configs import root_dirs

try:
    from symphony.data import input_pipeline_tf
    import tensorflow as tf

    tf.config.experimental.set_visible_devices([], "GPU")
except ImportError:
    logging.warning("TensorFlow not installed")


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


def name_from_workdir(workdir: str) -> str:
    """Returns the full name of the model from the workdir."""
    index = workdir.find("workdirs") + len("workdirs/")
    return workdir[index:]


def config_to_dataframe(config: ml_collections.ConfigDict) -> Dict[str, Any]:
    """Flattens a nested config into a Pandas dataframe."""

    # Compatibility with old configs.
    if "num_interactions" not in config:
        config.num_interactions = config.n_interactions
        del config.n_interactions

    if "num_channels" not in config:
        config.num_channels = config.n_atom_basis
        assert config.num_channels == config.n_filters
        del config.n_atom_basis, config.n_filters

    def iterate_with_prefix(dictionary: Dict[str, Any], prefix: str):
        """Iterates through a nested dictionary, yielding the flattened and prefixed keys and values."""
        for k, v in dictionary.items():
            if isinstance(v, dict):
                yield from iterate_with_prefix(v, prefix=f"{prefix}{k}.")
            else:
                yield prefix + k, v

    config_dict = dict(iterate_with_prefix(config.to_dict(), "config."))
    return pd.DataFrame().from_dict([config_dict])


def load_model_at_step(
    workdir: str, step: str, run_in_evaluation_mode: bool,
    res_alpha: Optional[float] = None, res_beta: Optional[float] = None
) -> Tuple[hk.Transformed, optax.Params, ml_collections.ConfigDict]:
    """Loads the model at a given step.

    This is a lightweight version of load_from_workdir, that only constructs the model and not the training state.
    """
    params_file = os.path.join(workdir, f"checkpoints/params_{step}.pkl")
    with open(params_file, "rb") as f:
        params = pickle.load(f)

    # # Remove the batch dimension, if it exists.
    # if "attempt6" in workdir and int(step) < 5000000:
    #     params = jax.tree_map(lambda x: x[0], params)

    with open(workdir + "/config.yml", "rt") as config_file:
        config = yaml.unsafe_load(config_file)
    assert config is not None
    config = ml_collections.ConfigDict(config)
    config.root_dir = root_dirs.get_root_dir(
        config.dataset, config.get("fragment_logic", "nn"), config.max_targets_per_graph
    )

    # Update config.
    if res_alpha is not None:
        logging.info(f"Setting res_alpha to {res_alpha}")
        config.target_position_predictor.res_alpha = res_alpha

    if res_beta is not None:
        logging.info(f"Setting res_beta to {res_beta}")
        config.target_position_predictor.res_beta = res_beta

    model = models.create_model(config, run_in_evaluation_mode=run_in_evaluation_mode)
    params = jax.tree_map(jnp.asarray, params)
    return model, params, config


def load_weighted_average_model_at_steps(
    workdir: str, steps: Sequence[str], run_in_evaluation_mode: bool
) -> Tuple[hk.Transformed, optax.Params, ml_collections.ConfigDict]:
    """Loads the model at given steps, and takes an equal average of the parameters."""
    for index, step in enumerate(steps):
        params_file = os.path.join(workdir, f"checkpoints/params_{step}.pkl")
        with open(params_file, "rb") as f:
            params = pickle.load(f)
        
        if index == 0:
            params_avg = params
        else:
            params_avg = jax.tree_map(lambda x, y: x + y, params_avg, params)
    params_avg = jax.tree_map(lambda x: x / len(steps), params_avg)

    with open(workdir + "/config.yml", "rt") as config_file:
        config = yaml.unsafe_load(config_file)
    assert config is not None
    config = ml_collections.ConfigDict(config)
    config.root_dir = root_dirs.get_root_dir(
        config.dataset, config.get("fragment_logic", "nn"), config.max_targets_per_graph
    )

    model = models.create_model(config, run_in_evaluation_mode=run_in_evaluation_mode)
    params_avg = jax.tree_map(jnp.asarray, params_avg)
    return model, params_avg, config



def get_results_as_dataframe(basedir: str) -> pd.DataFrame:
    """Returns the results for the given model as a pandas dataframe."""

    results = pd.DataFrame()
    for config_file_path in glob.glob(
        os.path.join(basedir, "**", "*.yml"), recursive=True
    ):
        workdir = os.path.dirname(config_file_path)
        try:
            config, best_state, _, metrics_for_best_state = load_from_workdir(workdir)
        except FileNotFoundError:
            logging.warning(f"Skipping {workdir} because it is incomplete.")
            continue

        num_params = sum(
            jax.tree_util.tree_leaves(jax.tree_map(jnp.size, best_state.params))
        )
        config_df = config_to_dataframe(config)
        other_df = pd.DataFrame.from_dict(
            {
                "model": [config.model.lower()],
                "max_l": [config.max_ell],
                "num_interactions": [config.num_interactions],
                "num_channels": [config.num_channels],
                "num_params": [num_params],
                # "num_train_molecules": [
                #     config.train_molecules[1] - config.train_molecules[0]
                # ],
            }
        )
        df = pd.merge(config_df, other_df, left_index=True, right_index=True)
        for split in metrics_for_best_state:
            metrics_for_split = {
                f"{split}.{metric}": [metrics_for_best_state[split][metric].item()]
                for metric in metrics_for_best_state[split]
            }
            metrics_df = pd.DataFrame.from_dict(metrics_for_split)
            df = pd.merge(df, metrics_df, left_index=True, right_index=True)
        results = pd.concat([results, df], ignore_index=True)

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
    config.root_dir = root_dirs.get_root_dir(
        config.dataset, config.get("fragment_logic", "nn"), config.max_targets_per_graph
    )

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
    config.root_dir = root_dirs.get_root_dir(
        config.dataset, config.get("fragment_logic", "nn"), config.max_targets_per_graph
    )

    # Mimic what we do in train.py.
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, dataset_rng = jax.random.split(rng)

    # Set up dummy variables to obtain the structure.
    rng, init_rng = jax.random.split(rng)

    net = models.create_model(config, run_in_evaluation_mode=False)
    eval_net = models.create_model(config, run_in_evaluation_mode=True)

    # If we have pickled parameters already, we don't need init_graphs to initialize the model.
    # Note that we restore the model parameters from the checkpoint anyways.
    # We only use the pickled parameters to initialize the model, so only the keys of the pickled parameters are important.
    if load_pickled_params:
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        pickled_params_file = os.path.join(checkpoint_dir, "params_best.pkl")
        if not os.path.exists(pickled_params_file):
            pickled_params_file = os.path.join(checkpoint_dir, "params_best.pkl")
            if not os.path.exists(pickled_params_file):
                raise FileNotFoundError(
                    f"No pickled params found at {pickled_params_file}"
                )

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


def construct_molecule(molecule_str: str) -> Tuple[ase.Atoms, str]:
    """Returns a molecule from the given input string.

    The input is interpreted either as an index for the QM9 dataset,
    a name for ase.build.molecule(),
    or a file with atomic numbers and coordinates for ase.io.read().
    """
    # If we believe the string is a file, try to read it.
    if os.path.exists(molecule_str):
        filename = os.path.basename(molecule_str).split(".")[0]
        return ase.io.read(molecule_str), filename

    # A number is interpreted as a QM9 molecule index.
    if molecule_str.isdigit():
        dataset = qm9.load_qm9("qm9_data")
        molecule = dataset[int(molecule_str)]
        return molecule, f"qm9_index={molecule_str}"

    # If the string is a valid molecule name, try to build it.
    molecule = ase.build.molecule(molecule_str)
    return molecule, molecule.get_chemical_formula()
