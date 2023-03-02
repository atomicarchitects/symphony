import logging

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import optax
import profile_nn_jax
from flax.training import train_state

import input_pipeline
import models
import qm9
import train

cutoff = 5.0  # Angstroms


@hk.without_apply_rng
@hk.transform
def net(graphs):
    lmax = 3
    output_irreps = 32 * e3nn.s2_irreps(lmax)
    return models.HaikuMACE(
        output_irreps=output_irreps,
        r_max=cutoff,
        num_interactions=2,
        hidden_irreps=output_irreps,
        readout_mlp_irreps=output_irreps,
        avg_num_neighbors=25.0,
        num_species=5,
        max_ell=3,
        position_coeffs_lmax=lmax,
    )(graphs)


def main():
    logging.basicConfig(level=logging.INFO)
    # profile_nn_jax.enable()

    atomic_numbers = jnp.array([1, 6, 7, 8, 9])

    print("Loading QM9 dataset...")
    molecules = qm9.load_qm9("qm9_data")

    dataset = input_pipeline.dataloader(
        jax.random.PRNGKey(0),
        molecules,
        atomic_numbers,
        nn_tolerance=0.125,
        nn_cutoff=cutoff,
        max_n_nodes=512,
        max_n_edges=1024,
        max_n_graphs=64,
    )
    print("Initialize model...")
    params = jax.jit(net.init)(jax.random.PRNGKey(1), next(dataset))
    apply_fn = jax.jit(net.apply)

    tx = optax.adam(learning_rate=0.001)
    state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)

    n_nodes = []
    n_edges = []
    norms = []

    print("Training...")
    for i in range(200):
        graphs = next(dataset)
        state, _metrics, grads_amp, grads_std = train.train_step(
            state,
            graphs,
            res_beta=60,
            res_alpha=99,
            radius_rbf_variance=(0.015) ** 2,
        )
        print(f"grads amp: {grads_amp:.2e} std: {grads_std:.2e}")
        # emb = apply_fn(params, graphs)
        # emb = emb.array[jraph.get_node_padding_mask(graphs), :]
        # graphs = jraph.unpad_with_graphs(graphs)
        # print(
        #     f"{graphs.n_node} {graphs.n_edge} "
        #     f"pred: [{emb.min():.2e}, {emb.max():.2e}] {jnp.linalg.norm(emb):.2e}"
        # )
        # n_nodes.append(graphs.n_node[0])
        # n_edges.append(graphs.n_edge[0])
        # norms.append(jnp.linalg.norm(emb))

    plt.plot(n_nodes, norms, "o")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
