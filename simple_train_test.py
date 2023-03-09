import logging
import pickle

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from flax.training import train_state

import input_pipeline
import models
import qm9
import train
import os

cutoff = 5.0  # Angstroms


@hk.without_apply_rng
@hk.transform
def net(graphs):
    lmax = 3
    output_irreps = 32 * e3nn.s2_irreps(lmax)
    return models.HaikuMACE(
        output_irreps=output_irreps,
        r_max=cutoff,
        num_interactions=3,
        hidden_irreps=output_irreps,
        readout_mlp_irreps=output_irreps,
        avg_num_neighbors=25.0,
        num_species=5,
        max_ell=3,
        position_coeffs_lmax=lmax,
    )(graphs)


def main():
    logging.basicConfig(level=logging.INFO)
    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_debug_infs", True)

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
        max_n_graphs=2,
    )
    if os.path.exists("params.pkl"):
        print("Loading model...")
        with open("params.pkl", "rb") as f:
            params = pickle.load(f)
    else:
        print("Initialize model...")
        params = jax.jit(net.init)(jax.random.PRNGKey(1), next(dataset))
        with open("params.pkl", "wb") as f:
            pickle.dump(params, f)

    apply_fn = jax.jit(net.apply)

    tx = optax.adam(learning_rate=0.001)
    state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)

    params_amps = []
    params_stds = []
    grads_amps = []
    grads_stds = []
    emb_amps = []
    emb_stds = []

    print("Training...")
    for i in range(200):
        graphs = next(dataset)

        # if i in [0]:
        #     print(f"skip step {i}")
        #     continue

        state, metrics, grads_amp, grads_std, emb_amp, emb_std = train.train_step(
            state,
            graphs,
            res_beta=60,
            res_alpha=99,
            radius_rbf_variance=(0.015) ** 2,
        )
        w = jnp.concatenate(
            [p.flatten() for p in jax.tree_util.tree_leaves(state.params)]
        )
        params_amp = jnp.max(jnp.abs(w))
        params_std = jnp.std(w)
        print(
            f"step {i}"
            f" grads amp: {grads_amp:.2e} std: {grads_std:.2e}"
            f" params amp: {params_amp:.2e} std: {params_std:.2e}"
            f" emb amp: {emb_amp:.2e} std: {emb_std:.2e}"
            f" loss: {metrics.total_loss}"
        )

        if jnp.isnan(params_amp):
            break

        params_amps.append(params_amp)
        params_stds.append(params_std)
        grads_amps.append(grads_amp)
        grads_stds.append(grads_std)
        emb_amps.append(emb_amp)
        emb_stds.append(emb_std)

    print(graphs)

    plt.plot(params_amps, label="params amp")
    plt.plot(params_stds, label="params std")
    plt.plot(grads_amps, label="grads amp")
    plt.plot(grads_stds, label="grads std")
    plt.plot(emb_amps, label="emb amp")
    plt.plot(emb_stds, label="emb std")
    plt.show()


if __name__ == "__main__":
    main()
