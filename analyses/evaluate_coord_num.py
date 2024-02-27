from typing import *
import ase
import ase.db
import ase.io
import functools
import jax
from jax import numpy as jnp
import jraph
import numpy as np
import os
import re
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from absl import flags, app

from analyses import analysis
from symphony import datatypes, models
from symphony.data import input_pipeline, input_pipeline_tf, tmqm

def device_batch(
    graph_iterator: Iterator[datatypes.Fragments],
) -> Iterator[datatypes.Fragments]:
    """Batches a set of graphs to the size of the number of devices."""
    num_devices = jax.local_device_count()
    batch = []
    for idx, graph in enumerate(graph_iterator):
        if idx % num_devices == num_devices - 1:
            batch.append(graph)
            batch = jax.tree_map(lambda *x: jnp.stack(x, axis=0), *batch)
            batch = datatypes.Fragments.from_graphstuple(batch)
            yield batch

            batch = []
        else:
            batch.append(graph)

@functools.partial(jax.pmap, axis_name="device", static_broadcasted_argnums=(1,))
def evaluate_model(graphs, apply_fn):
    rng = jax.random.PRNGKey(0)
    rng, pred_rng = jax.random.split(rng)
    num_graphs = graphs.n_node.shape[0]
    num_nodes = graphs.nodes.positions.shape[0]
    n_node = graphs.n_node
    segment_ids = models.get_segment_ids(n_node, num_nodes)

    pred = apply_fn(graphs, pred_rng)
    pred = jax.nn.softmax(pred)
    pred = jnp.round(pred)
    neighbor_preds = jraph.segment_sum(pred[:, 1], segment_ids, num_graphs)
    # just looking at predicting coordination # for now, as opposed to which specific atoms are neighbors
    correct = jax.lax.psum(neighbor_preds[:num_graphs] == jraph.segment_sum(graphs.nodes.neighbor_probs[:, 1], segment_ids, num_graphs),
            axis_name="device")
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
    #params_coord = {k:v for k, v in params.items() if "coord" in k}
    rng = jax.random.PRNGKey(0)
    rng, data_rng = jax.random.split(rng)
    mols_by_split = input_pipeline_tf.get_datasets(data_rng, config)
    apply_fn = jax.jit(lambda padded_fragment, rng: model.apply(
        params,
        rng,
        padded_fragment,
        1.0,
        1.0,
    ))

    for split in ['train', 'test']:
        dataset = mols_by_split[split]
        num_correct = 0
        num_mols = 0
        for graphs in device_batch(dataset.as_numpy_iterator()):
            num_correct += jnp.sum(evaluate_model(graphs, apply_fn))
            num_mols += jnp.ravel(graphs.n_node).shape[0]
            if num_mols >= 10000:
                break
        percent = num_correct / num_mols * 100
        print(f"{split}: {num_correct} / {num_mols} ({percent:.2f}%) coordination #s predicted correctly")


if __name__ == "__main__":
    flags.DEFINE_string(
        "workdir",
        "/data/NFS/potato/songk/spherical-harmonic-net/workdirs/",
        "Workdir for model.",
    )
    flags.DEFINE_string(
        "step",
        "best",
        "Step number to load model from. The default corresponds to the best model.",
    )
    app.run(main)
