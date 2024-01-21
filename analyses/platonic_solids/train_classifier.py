from typing import Optional, Callable
import functools

import tqdm
from absl import flags
from absl import app
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
import numpy as np
import os
import ase
import haiku as hk
import optax
import pickle
import jraph
import tensorflow as tf

from symphony.models import nequip
from symphony.data import input_pipeline_tf, input_pipeline

FLAGS = flags.FLAGS


class PlatonicSolidsClassifier(hk.Module):
    def __init__(
        self, node_embedder_fn: Callable[[], hk.Module], name: str | None = None
    ):
        super().__init__(name)
        self.num_pieces = 5
        self.node_embedder = node_embedder_fn()
        self.linear = hk.Linear(self.num_pieces)

    def __call__(self, graphs: jraph.GraphsTuple):
        node_feats = self.node_embedder(graphs)
        global_feats = e3nn.scatter_mean(node_feats, nel=graphs.n_node)
        # with jnp.printoptions(precision=2, suppress=True, formatter={'all':'{:0.2f}'.format}):
        #     jax.debug.print("0e={x}", x=jnp.abs(global_feats.filter("0e").array).mean(axis=-1).round(2))
        #     jax.debug.print("1o={x}", x=jnp.abs(global_feats.filter("1o").array).mean(axis=-1).round(2))
        #     jax.debug.print("2e={x}", x=jnp.abs(global_feats.filter("2e").array).mean(axis=-1).round(2))
        #     jax.debug.print("3o={x}", x=jnp.abs(global_feats.filter("3o").array).mean(axis=-1).round(2))
        #     jax.debug.print("4e={x}", x=jnp.abs(global_feats.filter("4e").array).mean(axis=-1).round(2))
        #     jax.debug.print("5o={x}", x=jnp.abs(global_feats.filter("5o").array).mean(axis=-1).round(2))
        #     jax.debug.print("6e={x}", x=jnp.abs(global_feats.filter("6e").array).mean(axis=-1).round(2))

        global_feats = global_feats.filter("0e").array
        global_feats = self.linear(global_feats)

        return global_feats


def create_model():
    @hk.without_apply_rng
    @hk.transform
    def model(graphs):
        classifier = PlatonicSolidsClassifier(
            node_embedder_fn=lambda: nequip.NequIP(
                num_species=1,
                r_max=3.0,
                avg_num_neighbors=1000.0,
                max_ell=6,
                init_embedding_dims=32,
                output_irreps=(32 * e3nn.s2_irreps(6)).regroup(),
                num_interactions=2,
                even_activation=jax.nn.swish,
                odd_activation=jax.nn.tanh,
                mlp_activation=jax.nn.swish,
                mlp_n_hidden=32,
                mlp_n_layers=2,
                n_radial_basis=8,
                skip_connection=True,
            ),
        )
        return classifier(graphs)

    return model


def get_graphs():
    """Get a set of batched graphs for the platonic solids."""
    pieces = input_pipeline_tf.get_pieces_for_platonic_solids()
    pieces_as_molecules = [
        ase.Atoms(numbers=np.asarray([1] * len(piece)), positions=np.asarray(piece))
        for piece in pieces
    ]
    pieces_as_graphs = [
        input_pipeline.ase_atoms_to_jraph_graph(molecule, [1], nn_cutoff=1.05)
        for molecule in pieces_as_molecules
    ]
    pieces_as_graphs = [
        piece._replace(globals=np.asarray([index], dtype=np.int32))
        for index, piece in enumerate(pieces_as_graphs)
    ]
    return jraph.batch(pieces_as_graphs)


@functools.partial(jax.jit, static_argnames=("model", "tx"))
def train_step(params, opt_state, rng, graphs, model, tx):
    def classification_loss(params):
        """Cross-entropy loss for classification."""
        labels = graphs.globals
        logits = model.apply(params, graphs)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()

    def compute_accuracy(params):
        """Compute accuracy."""
        labels = graphs.globals
        logits = model.apply(params, graphs)
        preds = jnp.argmax(logits, axis=-1)
        # jax.debug.print("preds={x}", x=preds)
        return jnp.mean(preds == labels)

    noisy_positions = graphs.nodes.positions + jax.random.normal(
        rng, graphs.nodes.positions.shape
    ) * 0.05
    graphs = graphs._replace(
        nodes=graphs.nodes._replace(
            positions=noisy_positions
        )
    )
    loss, grads = jax.value_and_grad(classification_loss)(params)
    accuracy = compute_accuracy(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    rng, _ = jax.random.split(rng)
    return params, opt_state, rng, loss, accuracy


def train_and_evaluate(
    num_train_steps: int,
    seed: int,
    outputdir: str,
):
    rng = jax.random.PRNGKey(seed)
    model = create_model()
    tx = optax.adam(1e-3)
    rng, init_rng = jax.random.split(rng)
    graphs = get_graphs()
    params = model.init(init_rng, graphs)
    opt_state = tx.init(params)

    with tqdm.trange(num_train_steps) as steps:
        for _ in steps:
            params, opt_state, rng, loss, accuracy = train_step(
                params, opt_state, rng, graphs, model, tx
            )
            steps.set_postfix(
                loss=f"{loss:.5f}",
                accuracy=f"{accuracy * 100:.1f}%",
            )

    # Save final parameters.
    os.makedirs(outputdir, exist_ok=True)
    with open(os.path.join(outputdir, "params.pkl"), "wb") as f:
        pickle.dump(params, f)


def main(unused_argv):
    del unused_argv

    train_and_evaluate(
        num_train_steps=FLAGS.num_train_steps,
        seed=FLAGS.seed,
        outputdir=FLAGS.outputdir,
    )


if __name__ == "__main__":
    flags.DEFINE_integer("seed", 0, "PRNG seed.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses", "platonic_solids", "classifiers", "nequip"),
        "Directory where molecules should be saved.",
    )
    flags.DEFINE_integer(
        "num_train_steps",
        1000,
        "Number of training steps.",
    )
    app.run(main)
