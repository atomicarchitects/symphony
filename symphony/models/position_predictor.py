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
        apply_gate_on_logits: bool,
        square_logits: bool,
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
        self.apply_gate_on_logits = apply_gate_on_logits
        self.square_logits = square_logits

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
        if self.apply_gate_on_logits:
            irreps = e3nn.Irreps(f"{self.position_coeffs_lmax}x0e") + s2_irreps
        else:
            irreps = s2_irreps

        log_position_coeffs = e3nn.haiku.Linear(
            self.num_radii * self.num_channels * irreps, force_irreps_out=True
        )(target_species_embeddings * focus_node_embeddings)
        log_position_coeffs = log_position_coeffs.mul_to_axis(factor=self.num_channels)
        log_position_coeffs = log_position_coeffs.mul_to_axis(factor=self.num_radii)

        # Apply the gate activation.
        if self.apply_gate_on_logits:
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

        if self.square_logits:
            position_logits = position_logits.apply(jnp.square)

        assert position_logits.shape == (
            num_graphs,
            self.num_radii,
            self.res_beta,
            self.res_alpha,
        )

        return log_position_coeffs, position_logits, angular_logits, radial_logits


class FactorizedTargetPositionPredictor(hk.Module):
    """Predicts the position coefficients for the target species, factorizing the radial and angular distributions."""

    def __init__(
        self,
        node_embedder: hk.Module,
        radial_flow: hk.Module,
        position_coeffs_lmax: int,
        res_beta: int,
        res_alpha: int,
        num_channels: int,
        num_species: int,
        num_radial_basis_fns: int,
        apply_gate_on_logits: bool,
        square_logits: bool,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.node_embedder = node_embedder
        self.radial_flow = radial_flow
        self.position_coeffs_lmax = position_coeffs_lmax
        self.res_beta = res_beta
        self.res_alpha = res_alpha
        self.num_channels = num_channels
        self.num_species = num_species
        self.num_radial_basis_fns = num_radial_basis_fns
        self.apply_gate_on_logits = apply_gate_on_logits
        self.square_logits = square_logits

    def create_radii(self) -> jnp.ndarray:
        """Creates the binned radii for the target positions."""
        return jnp.linspace(
            self.radial_flow.range_min,
            self.radial_flow.range_max,
            self.radial_flow.num_bins,
        )

    def compute_embeddings(
        self,
        graphs: datatypes.Fragments,
        focus_indices: jnp.ndarray,
        target_species: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        """Computes the node embeddings for the target positions."""
        num_graphs = graphs.n_node.shape[0]

        # Compute the focus node embeddings.
        node_embeddings = self.node_embedder(graphs)
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

        return focus_node_embeddings, target_species_embeddings

    def encode_radii(self, radii: jnp.ndarray, output_dims: int) -> jnp.ndarray:
        """Encodes the radii."""
        encoded_radii = e3nn.bessel(
            radii, n=self.num_radial_basis_fns, x_max=self.radial_flow.range_max
        )
        encoded_radii = e3nn.haiku.Linear(
            irreps_out=f"{output_dims}x0e",
        )(encoded_radii)
        return encoded_radii

    def predict_logits(
        self,
        graphs: datatypes.Fragments,
        focus_indices: jnp.ndarray,
        target_species: jnp.ndarray,
        true_radii: jnp.ndarray,
        inverse_temperature: float = 1.0,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.SphericalSignal]:
        # Compute the focus node embeddings and target species embeddings.
        focus_node_embeddings, target_species_embeddings = self.compute_embeddings(
            graphs, focus_indices, target_species
        )

        num_graphs = graphs.n_node.shape[0]

        # Predict the log-probs of the radii.
        conditioning = target_species_embeddings * focus_node_embeddings
        radial_logits = hk.vmap(self.radial_flow.log_prob, split_rng=False)(
            true_radii, conditioning.array
        )
        assert radial_logits.shape == (num_graphs, 1), radial_logits.shape

        # Encode the true radii, to condition the angular distribution on them.
        encoded_true_radii = self.encode_radii(
            true_radii, output_dims=focus_node_embeddings.irreps.num_irreps
        )

        # Predict the angular coefficients for the position signal.
        # These are actually describing the logits of the angular distribution.
        s2_irreps = e3nn.s2_irreps(self.position_coeffs_lmax, p_val=1, p_arg=-1)
        if self.apply_gate_on_logits:
            irreps = e3nn.Irreps(f"{self.position_coeffs_lmax}x0e") + s2_irreps
        else:
            irreps = s2_irreps

        # Compute the angular coefficients, conditioned on the target species,
        # focus node embeddings, and radii.
        log_angular_coeffs = e3nn.haiku.Linear(
            self.num_channels * irreps, force_irreps_out=True
        )(encoded_true_radii * conditioning)
        log_angular_coeffs = log_angular_coeffs.mul_to_axis(factor=self.num_channels)

        if self.apply_gate_on_logits:
            log_angular_coeffs = e3nn.gate(log_angular_coeffs)

        assert log_angular_coeffs.shape == (
            num_graphs,
            self.num_channels,
            s2_irreps.dim,
        )

        # Project onto a spherical grid.
        angular_logits = jax.vmap(
            lambda coeffs: utils.log_coeffs_to_logits(
                coeffs, self.res_beta, self.res_alpha, 1
            )
        )(
            log_angular_coeffs[:, :, None, :]
        )  # only one radius

        if self.square_logits:
            angular_logits = angular_logits.apply(jnp.square)

        # Scale the logits by the inverse temperature.
        angular_logits = angular_logits.apply(lambda val: val * inverse_temperature)

        return angular_logits, radial_logits

    def sample(
        self,
        graphs: datatypes.Fragments,
        focus_indices: jnp.ndarray,
        target_species: jnp.ndarray,
        inverse_temperature: float = 1.0,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.SphericalSignal]:
        # Compute the focus node embeddings and target species embeddings.
        focus_node_embeddings, target_species_embeddings = self.compute_embeddings(
            graphs, focus_indices, target_species
        )

        num_graphs = graphs.n_node.shape[0]

        # Predict the log-probs of the radii.
        conditioning = target_species_embeddings * focus_node_embeddings
        radii = hk.vmap(
            lambda condition: self.radial_flow.sample(condition, num_samples=1),
            split_rng=True,
        )(
            conditioning.array,
        )
        assert radii.shape == (num_graphs, 1), radii.shape

        # Encode the true radii, to condition the angular distribution on them.
        encoded_sampled_radii = self.encode_radii(
            radii, output_dims=focus_node_embeddings.irreps.num_irreps
        )

        # Predict the angular coefficients for the position signal.
        # These are actually describing the logits of the angular distribution.
        s2_irreps = e3nn.s2_irreps(self.position_coeffs_lmax, p_val=1, p_arg=-1)
        if self.apply_gate_on_logits:
            irreps = e3nn.Irreps(f"{self.position_coeffs_lmax}x0e") + s2_irreps
        else:
            irreps = s2_irreps

        # Compute the angular coefficients, conditioned on the target species,
        # focus node embeddings, and radii.
        log_angular_coeffs = e3nn.haiku.Linear(
            self.num_channels * irreps, force_irreps_out=True
        )(encoded_sampled_radii * conditioning)
        log_angular_coeffs = log_angular_coeffs.mul_to_axis(factor=self.num_channels)

        if self.apply_gate_on_logits:
            log_angular_coeffs = e3nn.gate(log_angular_coeffs)

        assert log_angular_coeffs.shape == (
            num_graphs,
            self.num_channels,
            s2_irreps.dim,
        )

        # Project onto a spherical grid.
        angular_logits = jax.vmap(
            lambda coeffs: utils.log_coeffs_to_logits(
                coeffs, self.res_beta, self.res_alpha, 1
            )
        )(
            log_angular_coeffs[:, :, None, :]
        )  # only one radius

        if self.square_logits:
            angular_logits = angular_logits.apply(jnp.square)

        # Scale the logits by the inverse temperature.
        angular_logits = angular_logits.apply(lambda val: val * inverse_temperature)

        # Convert to a distribution.
        angular_probs = jax.vmap(
            utils.position_logits_to_position_distribution,
        )(angular_logits)

        # Sample from angular distribution.
        angular_samples = jax.vmap(
            lambda probs: utils.sample_from_angular_distribution(probs)
        )(angular_probs)

        # Scale by the radii.
        position_vectors = radii * angular_samples

        return position_vectors
