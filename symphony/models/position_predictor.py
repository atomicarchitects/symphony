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

    def compute_node_embeddings(self, graphs: datatypes.Fragments) -> e3nn.IrrepsArray:
        """Computes the node embeddings for the target positions."""
        return self.node_embedder(graphs)

    def compute_focus_node_and_target_species_embeddings(
        self,
        graphs: datatypes.Fragments,
        focus_indices: jnp.ndarray,
        target_species: jnp.ndarray,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        """Computes the node embeddings for the target positions."""
        num_graphs = graphs.n_node.shape[0]

        # Compute the focus node embeddings.
        node_embeddings = self.compute_node_embeddings(graphs)
        focus_node_embeddings = node_embeddings[focus_indices]

        assert focus_node_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.dim,
        )

        # Compute the target species embeddings.
        target_species_embeddings = hk.Embed(
            self.num_species, embed_dim=focus_node_embeddings.irreps.num_irreps
        )(target_species)
        target_species_embeddings = e3nn.IrrepsArray(
            irreps=f"{focus_node_embeddings.irreps.num_irreps}x0e",
            array=target_species_embeddings,
        )

        assert target_species_embeddings.shape == (
            num_graphs,
            focus_node_embeddings.irreps.num_irreps,
        )

        return focus_node_embeddings, target_species_embeddings

    def compute_radial_conditioning(
        self,
        focus_node_embeddings: e3nn.IrrepsArray,
        target_species_embeddings: e3nn.IrrepsArray,
    ) -> e3nn.IrrepsArray:
        """Computes the radial conditioning."""
        # Apply linear projections to the original embeddings.
        focus_node_embeddings = e3nn.haiku.Linear(
            irreps_out=focus_node_embeddings.irreps,
        )(focus_node_embeddings)
        target_species_embeddings = e3nn.haiku.Linear(
            irreps_out=target_species_embeddings.irreps,
        )(target_species_embeddings)

        # Extract the scalars.
        target_species_embeddings_scalars = target_species_embeddings.filter(keep="0e")
        focus_node_embeddings_scalars = focus_node_embeddings.filter(keep="0e")
        all_scalars = e3nn.concatenate(
            [target_species_embeddings_scalars, focus_node_embeddings_scalars], axis=-1
        )
        return all_scalars

    def compute_angular_conditioning(
        self,
        focus_node_embeddings: e3nn.IrrepsArray,
        target_species_embeddings: e3nn.IrrepsArray,
        radii: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        """Computes the angular conditioning."""
        # Apply linear projections to the original embeddings.
        focus_node_embeddings = e3nn.haiku.Linear(
            irreps_out=focus_node_embeddings.irreps,
        )(focus_node_embeddings)
        target_species_embeddings = e3nn.haiku.Linear(
            irreps_out=target_species_embeddings.irreps,
        )(target_species_embeddings)

        # Combine with the radii.
        encoded_radii = e3nn.bessel(
            radii, n=self.num_radial_basis_fns, x_max=self.radial_flow.range_max
        )
        encoded_radii = e3nn.haiku.Linear(
            irreps_out=f"{focus_node_embeddings.irreps.num_irreps}x0e",
        )(encoded_radii)
        return encoded_radii * target_species_embeddings * focus_node_embeddings

    def predict_coeffs_for_angular_logits(
        self, angular_conditioning: e3nn.IrrepsArray
    ) -> e3nn.IrrepsArray:
        """Predicts the coefficients for the angular distribution."""
        num_graphs = angular_conditioning.shape[0]

        s2_irreps = e3nn.s2_irreps(self.position_coeffs_lmax, p_val=1, p_arg=-1)
        if self.apply_gate_on_logits:
            irreps = e3nn.Irreps(f"{self.position_coeffs_lmax}x0e") + s2_irreps
        else:
            irreps = s2_irreps

        log_angular_coeffs = e3nn.haiku.Linear(
            self.num_channels * irreps, force_irreps_out=True
        )(angular_conditioning)
        log_angular_coeffs = log_angular_coeffs.mul_to_axis(factor=self.num_channels)

        if self.apply_gate_on_logits:
            log_angular_coeffs = e3nn.gate(log_angular_coeffs)

        assert log_angular_coeffs.shape == (
            num_graphs,
            self.num_channels,
            s2_irreps.dim,
        )
        return log_angular_coeffs

    def angular_coeffs_to_logits(
        self, log_angular_coeffs: jnp.ndarray, inverse_temperature: float
    ) -> jnp.ndarray:
        """Converts the angular coefficients to logits."""

        # Project onto a spherical grid.
        angular_logits = jax.vmap(
            lambda coeffs: utils.log_coeffs_to_logits(
                coeffs, self.res_beta, self.res_alpha, 1
            )
        )(
            log_angular_coeffs[:, :, None, :]  # only one radius
        )

        # Remove the radial component.
        angular_logits = angular_logits[:, 0, :, :]

        if self.square_logits:
            angular_logits = angular_logits.apply(jnp.square)

        # Scale the logits by the inverse temperature.
        angular_logits = angular_logits.apply(lambda val: val * inverse_temperature)
        return angular_logits

    def compute_radii_pdf(self, conditioning: jnp.ndarray) -> jnp.ndarray:
        """Computes the probability density function of the radii."""
        all_radii = jnp.linspace(
            self.radial_flow.range_min, self.radial_flow.range_max, 1000
        )
        log_probs = hk.vmap(
            lambda radius: self.radial_flow.log_prob(radius, conditioning),
            split_rng=False,
        )(all_radii)
        return jax.nn.softmax(log_probs, axis=-1)

    def predict_logits(
        self,
        graphs: datatypes.Fragments,
        focus_indices: jnp.ndarray,
        target_species: jnp.ndarray,
        true_radii: jnp.ndarray,
        inverse_temperature: float = 1.0,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.SphericalSignal]:
        """Predicts the angular and radial logits for the target positions."""

        # Compute the focus node embeddings and target species embeddings.
        (
            focus_node_embeddings,
            target_species_embeddings,
        ) = self.compute_focus_node_and_target_species_embeddings(
            graphs, focus_indices, target_species
        )

        num_graphs = graphs.n_node.shape[0]

        # Predict the log-probs of the radii.
        radial_conditioning = self.compute_radial_conditioning(
            focus_node_embeddings, target_species_embeddings
        )
        radial_logits = hk.vmap(self.radial_flow.log_prob, split_rng=False)(
            true_radii, radial_conditioning
        )
        assert radial_logits.shape == (num_graphs,), radial_logits.shape

        # Compute the PDF of the radii.
        radii_pdf = hk.vmap(self.compute_radii_pdf, split_rng=False)(
            radial_conditioning
        )

        # Encode the true radii, to condition the angular distribution on them.
        angular_conditioning = self.compute_angular_conditioning(
            focus_node_embeddings, target_species_embeddings, true_radii
        )

        # Predict the coefficients for the angular distribution.
        log_angular_coeffs = self.predict_coeffs_for_angular_logits(
            angular_conditioning
        )

        # Project onto a spherical grid.
        angular_logits = self.angular_coeffs_to_logits(
            log_angular_coeffs, inverse_temperature
        )

        # Convert to a distribution.
        angular_probs = jax.vmap(
            utils.position_logits_to_position_distribution,
        )(angular_logits)

        return (
            radial_logits,
            log_angular_coeffs,
            angular_logits,
            angular_probs,
            radii_pdf,
        )

    def sample(
        self,
        graphs: datatypes.Fragments,
        focus_indices: jnp.ndarray,
        target_species: jnp.ndarray,
        inverse_temperature: float = 1.0,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.SphericalSignal]:
        """Samples the target positions."""
        # Compute the focus node embeddings and target species embeddings.
        (
            focus_node_embeddings,
            target_species_embeddings,
        ) = self.compute_focus_node_and_target_species_embeddings(
            graphs, focus_indices, target_species
        )

        num_graphs = graphs.n_node.shape[0]

        # Sample the radii.
        # Also measure the log-probs of the sampled radii.
        radial_conditioning = self.compute_radial_conditioning(
            focus_node_embeddings, target_species_embeddings
        )
        sampled_radii = hk.vmap(
            lambda condition: self.radial_flow.sample(condition),
            split_rng=True,
        )(
            radial_conditioning,
        )
        radial_logits = hk.vmap(self.radial_flow.log_prob, split_rng=False)(
            sampled_radii, radial_conditioning
        )
        assert sampled_radii.shape == (num_graphs,), sampled_radii.shape

        # Compute the PDF of the radii.
        radii_pdf = hk.vmap(self.compute_radii_pdf, split_rng=False)(
            radial_conditioning
        )

        # Encode the sampled radii, to condition the angular distribution on them.
        # sampled_radii = jnp.ones_like(sampled_radii)
        angular_conditioning = self.compute_angular_conditioning(
            focus_node_embeddings, target_species_embeddings, sampled_radii
        )

        # Predict the coefficients for the angular distribution.
        log_angular_coeffs = self.predict_coeffs_for_angular_logits(
            angular_conditioning
        )

        # Project onto a spherical grid.
        angular_logits = self.angular_coeffs_to_logits(
            log_angular_coeffs, inverse_temperature
        )

        # Convert to a distribution.
        angular_probs = jax.vmap(
            utils.position_logits_to_position_distribution,
        )(angular_logits)

        # Sample from angular distribution.
        rng = hk.next_rng_key()
        rngs = jax.random.split(rng, num_graphs)
        angular_samples = jax.vmap(
            lambda probs, rng: utils.sample_from_angular_distribution(probs, rng)
        )(angular_probs, rngs)

        # Scale by the radii.
        position_vectors = sampled_radii[:, None] * angular_samples

        return (
            position_vectors,
            radial_logits,
            sampled_radii,
            log_angular_coeffs,
            angular_logits,
            angular_probs,
            radii_pdf,
        )
