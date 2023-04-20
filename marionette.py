# An architecture pulling the best out of
# - NEQUIP simplicity
# - MACE polynomial structure
# - ESCN performance


from typing import Callable, Optional, Union

import e3nn_jax as e3nn
import flax
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax.experimental.linear_shtp import LinearSHTP


class MarioNetteLayerFlax(flax.linen.Module):
    avg_num_neighbors: float
    num_species: int = 1
    output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e")
    interaction_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e")
    soft_normalization: float = 1e5
    even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu
    odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh
    mlp_n_hidden: int = 64
    mlp_n_layers: int = 2
    n_radial_basis: int = 8

    @flax.linen.compact
    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        node_feats: e3nn.IrrepsArray,
        node_specie: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ):
        return _impl(
            e3nn.flax.Linear,
            e3nn.flax.MultiLayerPerceptron,
            self,
            vectors,
            node_feats,
            node_specie,
            senders,
            receivers,
        )


class MarioNetteLayerHaiku(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        num_species: int = 1,
        output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e"),
        interaction_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e"),
        soft_normalization: float = 1e5,
        even_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        odd_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.tanh,
        mlp_n_hidden: int = 64,
        mlp_n_layers: int = 2,
        n_radial_basis: int = 8,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.avg_num_neighbors = avg_num_neighbors
        self.num_species = num_species
        self.output_irreps = output_irreps
        self.interaction_irreps = interaction_irreps
        self.soft_normalization = soft_normalization
        self.even_activation = even_activation
        self.odd_activation = odd_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.n_radial_basis = n_radial_basis

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        node_feats: e3nn.IrrepsArray,
        node_specie: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ):
        return _impl(
            e3nn.haiku.Linear,
            e3nn.haiku.MultiLayerPerceptron,
            self,
            vectors,
            node_feats,
            node_specie,
            senders,
            receivers,
        )


def _impl(
    Linear: Callable,
    MultiLayerPerceptron: Callable,
    self: Union[MarioNetteLayerFlax, MarioNetteLayerHaiku],
    vectors: e3nn.IrrepsArray,  # [n_edges, 3]
    node_feats: e3nn.IrrepsArray,  # [n_nodes, irreps]
    node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
    senders: jnp.ndarray,  # [n_edges]
    receivers: jnp.ndarray,  # [n_edges]
):
    n_edge = vectors.shape[0]
    n_node = node_feats.shape[0]
    assert vectors.shape == (n_edge, 3)
    assert node_feats.shape == (n_node, node_feats.irreps.dim)
    assert node_specie.shape == (n_node,)
    assert senders.shape == (n_edge,)
    assert receivers.shape == (n_edge,)

    interaction_irreps = e3nn.Irreps(self.interaction_irreps)
    output_irreps = e3nn.Irreps(self.output_irreps)

    # Self connection
    self_connection = Linear(
        output_irreps, num_indexed_weights=self.num_species, name="skip_tp"
    )(
        node_specie, node_feats
    )  # [n_nodes, output_irreps]

    node_feats = Linear(node_feats.irreps, name="linear_up")(node_feats)

    messages = node_feats[senders]

    conv = LinearSHTP(interaction_irreps, mix=False)
    w_unused = conv.init(jax.random.PRNGKey(0), messages[0], vectors[0])
    w_unused_flat = flatten(w_unused)

    # Radial part
    lengths = e3nn.norm(vectors).array  # [n_edges, 1]
    mix = MultiLayerPerceptron(
        self.mlp_n_layers * (self.mlp_n_hidden,) + (w_unused_flat.size,),
        self.mlp_activation,
        output_activation=False,
    )(
        e3nn.bessel(lengths[:, 0], self.n_radial_basis) * e3nn.soft_envelope(lengths),
    )  # [n_edges, w_unused_flat.size]

    # Discard 0 length edges that come from graph padding
    mix = jnp.where(lengths == 0.0, 0.0, mix)

    # vmap over edges
    w = jax.vmap(unflatten, (0, None))(mix, w_unused)
    messages = jax.vmap(conv.apply)(w, messages, vectors)

    # Message passing
    zeros = e3nn.IrrepsArray.zeros(
        messages.irreps, node_feats.shape[:1], messages.dtype
    )
    node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]
    node_feats = node_feats / jnp.sqrt(self.avg_num_neighbors)

    node_feats = Linear(interaction_irreps, name="linear_down")(node_feats)

    # Activation
    node_feats = activation(node_feats, self.even_activation, self.odd_activation)
    node_feats = Linear(output_irreps, name="linear_out")(node_feats)

    # Soft normalization
    node_feats = soft_normalization(node_feats, self.soft_normalization)

    node_feats = 0.9 * self_connection + 0.45 * node_feats  # [n_nodes, irreps]

    assert node_feats.irreps == output_irreps
    assert node_feats.shape == (n_node, output_irreps.dim)
    return node_feats


def activation(
    x: e3nn.IrrepsArray, even_activation, odd_activation
) -> e3nn.IrrepsArray:
    x = e3nn.scalar_activation(
        x,
        [
            {1: even_activation, -1: odd_activation}[ir.p] if ir.l == 0 else None
            for _, ir in x.irreps
        ],
    )
    x = e3nn.concatenate(
        [
            x,
            e3nn.tensor_square(x.mul_to_axis()).axis_to_mul(),
        ]
    )
    return x


def soft_normalization(x: e3nn.IrrepsArray, max_norm: float = 1.0) -> e3nn.IrrepsArray:
    def phi(n):
        n = n / max_norm
        return 1.0 / (1.0 + n * e3nn.sus(n))

    return e3nn.norm_activation(x, [phi] * len(x.irreps))


def flatten(w):
    return jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(w)])


def unflatten(array, template):
    lst = []
    start = 0
    for x in jax.tree_util.tree_leaves(template):
        lst.append(array[start : start + x.size].reshape(x.shape))
        start += x.size
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(template), lst)
