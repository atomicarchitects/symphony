from typing import Callable, Optional, Tuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from symphony import datatypes
from symphony.models import utils, sphericalconv


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

    def compute_node_embeddings(self, graphs: datatypes.Fragments) -> e3nn.IrrepsArray:
        """Computes the node embeddings for the target positions."""
        return self.node_embedder(graphs)

    def __call__(
        self,
        graphs: datatypes.Fragments,
        focus_indices: jnp.ndarray,
        target_species: jnp.ndarray,
        inverse_temperature: float = 1.0,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.SphericalSignal]:
        num_graphs = graphs.n_node.shape[0]

        # Compute the focus node embeddings.
        node_embeddings = self.compute_node_embeddings(graphs)
        focus_node_embeddings = node_embeddings[focus_indices]

        assert focus_node_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.dim,
        )

        target_species_embeddings = hk.Embed(
            self.num_species, embed_dim=focus_node_embeddings.irreps.num_irreps
        )(target_species)

        assert target_species_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.num_irreps,
        )

        # Create the irreps for projecting onto the spherical harmonics.
        # Also, add a few scalars for the gate activation.
        s2_irreps = e3nn.s2_irreps(self.position_coeffs_lmax, p_val=1, p_arg=-1)
        if self.apply_gate:
            irreps = e3nn.Irreps(f"{self.position_coeffs_lmax}x0e") + s2_irreps
        else:
            irreps = s2_irreps

        log_position_coeffs = e3nn.haiku.Linear(
            self.num_radii * self.num_channels * irreps, force_irreps_out=True
        )(target_species_embeddings * focus_node_embeddings)
        # log_position_coeffs = sphericalconv.SphericalConvolution(
        #     self.res_beta,
        #     self.res_alpha,
        #     log_position_coeffs.irreps.lmax,
        #     self.num_channels,
        #     self.num_channels,
        #     jax.nn.softplus,
        # )(log_position_coeffs)
        log_position_coeffs = log_position_coeffs.mul_to_axis(factor=self.num_channels)
        log_position_coeffs = log_position_coeffs.mul_to_axis(factor=self.num_radii)

        # Apply the gate activation.
        if self.apply_gate:
            log_position_coeffs = e3nn.gate(log_position_coeffs)

        assert log_position_coeffs.shape == (
            num_graphs,
            self.num_channels,
            self.num_radii,
            s2_irreps.dim,
        )

        # Not relevant for the unfactorized case.
        angular_logits, radial_logits = None, None

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

        return log_position_coeffs, position_logits, angular_logits, radial_logits


class FactorizedTargetPositionPredictor(hk.Module):
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
        radial_mlp_latent_size: int,
        radial_mlp_num_layers: int,
        radial_mlp_activation: Callable[[jnp.ndarray], jnp.ndarray],
        apply_gate: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.node_embedder = node_embedder
        self.position_coeffs_lmax = position_coeffs_lmax
        self.res_beta = res_beta
        self.res_alpha = res_alpha
        self.radial_mlp_latent_size = radial_mlp_latent_size
        self.radial_mlp_num_layers = radial_mlp_num_layers
        self.radial_mlp_activation = radial_mlp_activation
        self.num_channels = num_channels
        self.num_species = num_species
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_radii = num_radii
        self.apply_gate = apply_gate

    def create_radii(self) -> jnp.ndarray:
        """Creates the binned radii for the target positions."""
        return jnp.linspace(self.min_radius, self.max_radius, self.num_radii)

    def compute_node_embeddings(self, graphs: datatypes.Fragments) -> e3nn.IrrepsArray:
        """Computes the node embeddings for the target positions."""
        return self.node_embedder(graphs)

    def __call__(
        self,
        graphs: datatypes.Fragments,
        focus_indices: jnp.ndarray,
        target_species: jnp.ndarray,
        inverse_temperature: float = 1.0,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.SphericalSignal]:
        num_graphs = graphs.n_node.shape[0]

        # Compute the focus node embeddings.
        node_embeddings = self.compute_node_embeddings(graphs)
        focus_node_embeddings = node_embeddings[focus_indices]

        assert focus_node_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.dim,
        )

        target_species_embeddings = hk.Embed(
            self.num_species, embed_dim=focus_node_embeddings.irreps.num_irreps
        )(target_species)

        assert target_species_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.num_irreps,
        )

        # Predict the radii for the target positions.
        radial_logits = e3nn.haiku.Linear(f"{self.num_radii}x0e")(
            target_species_embeddings * focus_node_embeddings
        )
        radial_logits = e3nn.haiku.MultiLayerPerceptron(
            list_neurons=[self.radial_mlp_latent_size]
            * (self.radial_mlp_num_layers - 1)
            + [self.num_radii],
            act=self.radial_mlp_activation,
            output_activation=False,
        )(radial_logits).array
        assert radial_logits.shape == (num_graphs, self.num_radii)

        # Predict the angular coefficients for the position signal.
        # These are actually describing the logits of the angular distribution.
        s2_irreps = e3nn.s2_irreps(self.position_coeffs_lmax, p_val=1, p_arg=-1)
        if self.apply_gate:
            irreps = e3nn.Irreps(f"{self.position_coeffs_lmax}x0e") + s2_irreps
        else:
            irreps = s2_irreps

        log_angular_coeffs = e3nn.haiku.Linear(
            self.num_channels * irreps, force_irreps_out=True
        )(target_species_embeddings * focus_node_embeddings)
        log_angular_coeffs = log_angular_coeffs.mul_to_axis(factor=self.num_channels)

        if self.apply_gate:
            log_angular_coeffs = e3nn.gate(log_angular_coeffs)

        assert log_angular_coeffs.shape == (
            num_graphs,
            self.num_channels,
            s2_irreps.dim,
        )

        angular_logits = jax.vmap(
            lambda coeffs: utils.log_coeffs_to_logits(
                coeffs, self.res_beta, self.res_alpha, 1
            )
        )(
            log_angular_coeffs[:, :, None, :]
        )  # only one radius

        # Mix the radial components with each channel of the angular components.
        log_position_coeffs = jax.vmap(
            jax.vmap(
                utils.compute_coefficients_of_logits_of_joint_distribution,
                in_axes=(None, 0),
            )
        )(radial_logits, log_angular_coeffs)

        assert log_position_coeffs.shape == (
            num_graphs,
            self.num_channels,
            self.num_radii,
            s2_irreps.dim,
        )

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

        return log_position_coeffs, position_logits, angular_logits, radial_logits
