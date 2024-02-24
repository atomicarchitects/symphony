from typing import *
import ase
import ase.db
import ase.io
import jax
from jax import numpy as jnp
import jraph
import numpy as np
import os
import re
import sys
import tensorflow as tf

from absl import flags, app

from symphony import analysis, models
from symphony.data import input_pipeline, input_pipeline_tf, tmqm

def evaluate_model(apply_fn, dataset):
    correct = 0
    rng = jax.random.PRNGKey(0)
    for graphs in dataset.as_numpy_iterator():
        rng, pred_rng = jax.random.split(rng)
        num_graphs = graphs.n_node.shape[0]
        num_nodes = graphs.nodes.positions.shape[0]
        n_node = graphs.n_node
        segment_ids = models.get_segment_ids(n_node, num_nodes)

        pred = apply_fn(graphs, pred_rng)
        pred = jnp.round(pred)
        neighbor_preds = jraph.segment_sum(pred, segment_ids, num_graphs)
        neighbor_targets = jraph.segment_sum(graphs.nodes.neighbor_probs, segment_ids, num_graphs)
        correct += jnp.sum(jnp.where(neighbor_preds == graphs.nodes.neighbor_probs), 1, 0)
    return correct

def main(unused_argv: Sequence[str]):
    workdir = flags.FLAGS.workdir
    step = flags.FLAGS.step

    splits = {
        'train': (0, 8000),
        'val': (8000, 9000),
        'test': (9000, 10000)
    }
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True,
    )
    mols_by_split = input_pipeline_tf.get_datasets(config)
    padded_fragment = mols_by_split['val'].as_numpy_iterator().next()
    apply_fn = model.apply(
        params,
        jax.PRNGKey(0),
        padded_fragment,
        1.0,
        1.0,
    )

    for split in ['train', 'test']:
        dataset = mols_by_split[split]
        num_mols = len(splits[split])
        num_correct = evaluate_model(apply_fn, dataset)
        print(f"{split}: {num_correct} / {num_mols} ({num_correct / num_mols * 100 : .2f}%) coordination #s predicted correctly")


if __name__ == "__main__":
    flags.DEFINE_string(
        "workdir",
        "/data/NFS/potato/songk/spherical-harmonic-net/workdirs/",
        "Workdir for model.",
    )
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "conditional_generation", "analysed_workdirs"),
        "Directory where molecules should be saved.",
    )
    flags.DEFINE_bool(
        "visualize",
        False,
        "Whether to visualize the generation process step-by-step.",
    )
    flags.DEFINE_string(
        "step",
        "best",
        "Step number to load model from. The default corresponds to the best model.",
    )
    app.run(main)
