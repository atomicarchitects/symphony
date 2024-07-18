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
