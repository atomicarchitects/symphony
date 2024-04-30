from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from symphony import datatypes
from symphony.models import utils


class TargetPositionPredictor(hk.Module):
    """Predicts the position coefficients for the target species."""

    def __init__(
        self,
        node_embedder: hk.Module,
        position_coeffs_lmax: int,
        res_beta: int,
        res_alpha: int,
        num_channels: int,
        num_species: int,
        min_radius: float,
        max_radius: float,
        num_radii: int,
        apply_gate: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.node_embedder = node_embedder
        self.position_coeffs_lmax = position_coeffs_lmax
        self.res_beta = res_beta
        self.res_alpha = res_alpha
        self.num_channels = num_channels
        self.num_species = num_species
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_radii = num_radii
        self.apply_gate = apply_gate

    def create_radii(self) -> jnp.ndarray:
        """Creates the binned radii for the target positions."""
        return jnp.linspace(self.min_radius, self.max_radius, self.num_radii)

    def __call__(
        self,
        graphs: datatypes.Fragments,
        target_species: jnp.ndarray,
        inverse_temperature: float = 1.0,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.SphericalSignal]:
        num_graphs = graphs.n_node.shape[0]
        num_nodes = graphs.nodes.positions.shape[0]
        num_nodes_for_multifocus = graphs.globals.target_positions.shape[1]

        # Compute the focus node embeddings.
        node_embeddings = self.node_embedder(graphs)
        focus_node_embeddings = node_embeddings[focus_indices]

        assert focus_node_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.dim,
        )

        target_species_embeddings = hk.Embed(
            self.num_species, embed_dim=node_embeddings.irreps.num_irreps
        )(target_species)

        assert target_species_embeddings.shape == (
            num_graphs,
            num_nodes_for_multifocus,
            node_embeddings.irreps.num_irreps,
        ), print(target_species_embeddings.shape)

        # Create the irreps for projecting onto the spherical harmonics.
        # Also, add a few scalars for the gate activation.
        s2_irreps = e3nn.s2_irreps(self.position_coeffs_lmax, p_val=1, p_arg=-1)
        if self.apply_gate:
            irreps = e3nn.Irreps(f"{self.position_coeffs_lmax}x0e") + s2_irreps
        else:
            irreps = s2_irreps

        focus_mask_2 = jnp.where(jnp.arange(num_nodes) * graphs.nodes.focus_mask != 0, size=num_nodes_for_multifocus)[0]
        log_position_coeffs = e3nn.haiku.Linear(
            self.num_radii * self.num_channels * irreps, force_irreps_out=True
        )(target_species_embeddings * node_embeddings[focus_mask_2])  # TODO
        log_position_coeffs = log_position_coeffs.mul_to_axis(factor=self.num_channels)
        log_position_coeffs = log_position_coeffs.mul_to_axis(factor=self.num_radii)

        # Apply the gate activation.
        if self.apply_gate:
            log_position_coeffs = e3nn.gate(log_position_coeffs)

        assert log_position_coeffs.shape == (
            num_graphs,
            num_nodes_for_multifocus,
            self.num_channels,
            self.num_radii,
            s2_irreps.dim,
        ), print(log_position_coeffs.shape)

        # Scale the coefficients of logits by the inverse temperature.
        log_position_coeffs = log_position_coeffs * inverse_temperature

        # Convert the coefficients to a signal on the grid.
        position_logits = jax.vmap(
            lambda coeffs: utils.log_coeffs_to_logits(
                coeffs, self.res_beta, self.res_alpha, self.num_radii
            )
        )(log_position_coeffs)
        assert position_logits.shape == (
            num_graphs,
            self.num_radii,
            self.res_beta,
            self.res_alpha,
        )

        return log_position_coeffs, position_logits
