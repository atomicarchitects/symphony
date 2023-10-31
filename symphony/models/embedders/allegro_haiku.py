### Haiku version of https://github.com/mariogeiger/allegro-jax


from typing import Callable, List, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp


def normalized_bessel(d: jnp.ndarray, n: int) -> jnp.ndarray:
    with jax.ensure_compile_time_eval():
        r = jnp.linspace(0.0, 1.0, 1000, dtype=d.dtype)
        b = e3nn.bessel(r, n)
        mu = jnp.trapz(b, r, axis=0)
        sig = jnp.trapz((b - mu) ** 2, r, axis=0) ** 0.5
    return (e3nn.bessel(d, n) - mu) / sig


def u(d: jnp.ndarray, p: int) -> jnp.ndarray:
    return e3nn.poly_envelope(p - 1, 2)(d)


class AllegroLayer(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        max_ell: int = 3,
        output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e"),
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
        mlp_n_hidden: int = 64,
        mlp_n_layers: int = 3,
        p: int = 6,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.avg_num_neighbors = avg_num_neighbors
        self.max_ell = max_ell
        self.output_irreps = output_irreps
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.p = p

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        x: jnp.ndarray,  # [n_edges, features]
        V: e3nn.IrrepsArray,  # [n_edges, irreps]
        senders: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert x.shape == (num_edges, x.shape[-1])
        assert V.shape == (num_edges, V.irreps.dim)
        assert senders.shape == (num_edges,)

        irreps_out = e3nn.Irreps(self.output_irreps)

        w = e3nn.haiku.MultiLayerPerceptron((V.irreps.mul_gcd,), act=None)(x)
        Y = e3nn.spherical_harmonics(range(self.max_ell + 1), vectors, True)
        wY = e3nn.scatter_sum(
            w[:, :, None] * Y[:, None, :], dst=senders, map_back=True
        ) / jnp.sqrt(self.avg_num_neighbors)
        assert wY.shape == (num_edges, V.irreps.mul_gcd, wY.irreps.dim)

        V = e3nn.tensor_product(
            wY, V.mul_to_axis(), filter_ir_out="0e" + irreps_out
        ).axis_to_mul()

        if "0e" in V.irreps:
            x = jnp.concatenate([x, V.filter(keep="0e").array], axis=1)
            V = V.filter(drop="0e")

        x = e3nn.haiku.MultiLayerPerceptron(
            (self.mlp_n_hidden,) * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False,
        )(x)
        lengths = e3nn.norm(vectors).array
        x = u(lengths, self.p) * x
        assert x.shape == (num_edges, self.mlp_n_hidden)

        V = e3nn.haiku.Linear(irreps_out)(V)
        assert V.shape == (num_edges, V.irreps.dim)

        return (x, V)


def filter_layers(layer_irreps: List[e3nn.Irreps], max_ell: int) -> List[e3nn.Irreps]:
    layer_irreps = list(layer_irreps)
    filtered = [e3nn.Irreps(layer_irreps[-1])]
    for irreps in reversed(layer_irreps[:-1]):
        irreps = e3nn.Irreps(irreps)
        irreps = irreps.filter(
            keep=e3nn.tensor_product(
                filtered[0],
                e3nn.Irreps.spherical_harmonics(lmax=max_ell),
            ).regroup()
        )
        filtered.insert(0, irreps)
    return filtered


class Allegro(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        max_ell: int = 3,
        irreps: e3nn.Irreps = 128 * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e"),
        mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
        mlp_n_hidden: int = 1024,
        mlp_n_layers: int = 3,
        p: int = 6,
        n_radial_basis: int = 8,
        radial_cutoff: float = 1.0,
        output_irreps: e3nn.Irreps = e3nn.Irreps("0e"),
        num_layers: int = 3,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.avg_num_neighbors = avg_num_neighbors
        self.max_ell = max_ell
        self.irreps = irreps
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.p = p
        self.n_radial_basis = n_radial_basis
        self.radial_cutoff = radial_cutoff
        self.output_irreps = output_irreps
        self.num_layers = num_layers

    def __call__(
        self,
        node_attrs: jnp.ndarray,  # jax.nn.one_hot(z, num_species)
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        edge_feats: Optional[e3nn.IrrepsArray] = None,  # [n_edges, irreps]
    ) -> e3nn.IrrepsArray:
        num_edges = vectors.shape[0]
        num_nodes = node_attrs.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert node_attrs.shape == (num_nodes, node_attrs.shape[-1])
        assert senders.shape == (num_edges,)
        assert receivers.shape == (num_edges,)

        assert vectors.irreps in ["1o", "1e"]
        irreps = e3nn.Irreps(self.irreps)
        irreps_out = e3nn.Irreps(self.output_irreps)

        irreps_layers = [irreps] * self.num_layers + [irreps_out]
        irreps_layers = filter_layers(irreps_layers, self.max_ell)

        vectors = vectors / self.radial_cutoff

        d = e3nn.norm(vectors).array.squeeze(1)
        x = jnp.concatenate(
            [
                normalized_bessel(d, self.n_radial_basis),
                node_attrs[senders],
                node_attrs[receivers],
            ],
            axis=1,
        )
        assert x.shape == (num_edges, self.n_radial_basis + 2 * node_attrs.shape[-1])

        # Protection against exploding dummy edges:
        x = jnp.where(d[:, None] == 0.0, 0.0, x)

        x = e3nn.haiku.MultiLayerPerceptron(
            (
                self.mlp_n_hidden // 8,
                self.mlp_n_hidden // 4,
                self.mlp_n_hidden // 2,
                self.mlp_n_hidden,
            ),
            self.mlp_activation,
            output_activation=False,
        )(x)
        x = u(d, self.p)[:, None] * x
        assert x.shape == (num_edges, self.mlp_n_hidden)

        irreps_Y = irreps_layers[0].filter(
            keep=lambda mir: vectors.irreps[0].ir.p ** mir.ir.l == mir.ir.p
        )
        V = e3nn.spherical_harmonics(irreps_Y, vectors, True)

        if edge_feats is not None:
            V = e3nn.concatenate([V, edge_feats])

        w = e3nn.haiku.MultiLayerPerceptron((V.irreps.num_irreps,), act=None)(x)
        V = w * V
        assert V.shape == (num_edges, V.irreps.dim)

        for irreps in irreps_layers[1:]:
            y, V = AllegroLayer(
                avg_num_neighbors=self.avg_num_neighbors,
                max_ell=self.max_ell,
                output_irreps=irreps,
                mlp_activation=self.mlp_activation,
                mlp_n_hidden=self.mlp_n_hidden,
                mlp_n_layers=self.mlp_n_layers,
                p=self.p,
            )(vectors, x, V, senders)

            alpha = 0.5
            x = (x + alpha * y) / jnp.sqrt(1 + alpha**2)

        x = e3nn.haiku.MultiLayerPerceptron((128,), act=None)(x)

        xV = e3nn.haiku.Linear(irreps_out)(e3nn.concatenate([x, V]))

        if xV.irreps != irreps_out:
            raise ValueError(f"output_irreps {irreps_out} is not reachable")

        return xV
