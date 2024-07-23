from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import e3nn_jax as e3nn

from symphony import datatypes
from symphony.models.utils import utils
from symphony.models.angular_predictors import AngularPredictor
from symphony.models.radius_predictors import RadiusPredictor


class TargetPositionPredictor(hk.Module):
    """Predicts the position coefficients for the target species."""

    def __init__(
        self,
        node_embedder_fn: Callable[[], hk.Module],
        radial_predictor_fn: Callable[[], RadiusPredictor],
        angular_predictor_fn: Callable[[], AngularPredictor],
        num_species: int,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.node_embedder = node_embedder_fn()
        self.radial_predictor = radial_predictor_fn()
        self.angular_predictor = angular_predictor_fn()
        self.num_species = num_species

    def compute_conditioning(
        self,
        graphs: datatypes.Fragments,
        target_species: jnp.ndarray,
        num_nodes_for_multifocus: int,
        focus_indices: jnp.ndarray,
        focus_mask: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        """Computes the conditioning for the target position predictor."""
        num_graphs = graphs.n_node.shape[0]
        cum_num_nodes = jnp.concatenate([jnp.zeros(1), jnp.cumsum(graphs.n_node)])
        num_nodes = graphs.nodes.positions.shape[0]

        # Compute the focus node embeddings.
        node_embeddings = self.node_embedder(graphs)

        def get_focus_embeddings(indices: jnp.ndarray, mask: jnp.ndarray, segment_id: int):
            indices_shifted = indices + cum_num_nodes[segment_id]
            return e3nn.IrrepsArray(  # TODO: is there no element-wise multiplication in e3nn?
                node_embeddings.irreps,
                node_embeddings.array[indices_shifted.astype(jnp.int32)] * mask[:, None]
            )
        focus_node_embeddings = jax.vmap(
            get_focus_embeddings,
        )(focus_indices, focus_mask, jnp.arange(num_graphs))

        assert focus_node_embeddings.shape == (
            num_graphs,
            num_nodes_for_multifocus,
            focus_node_embeddings.irreps.dim,
        ), (
            focus_node_embeddings.shape,
            focus_node_embeddings.irreps.dim,
        )

        # Embed the target species.
        target_species_embeddings = hk.Embed(
            self.num_species,
            embed_dim=focus_node_embeddings.irreps.num_irreps,
        )(target_species)
        assert target_species_embeddings.shape == (
            num_graphs,
            num_nodes_for_multifocus,
            focus_node_embeddings.irreps.num_irreps,
        ), (
            target_species_embeddings.shape,
            target_species.shape,
            focus_node_embeddings.shape,
            self.num_species,
            (num_graphs, focus_node_embeddings.irreps.num_irreps),
        )

        # Concatenate the focus and target species embeddings.
        conditioning = e3nn.concatenate(
            [focus_node_embeddings, target_species_embeddings], axis=-1
        )
        assert conditioning.shape == (
            num_graphs, 
            num_nodes_for_multifocus,
            conditioning.irreps.dim
        )
        return conditioning

    def get_training_predictions(
        self,
        graphs: datatypes.Fragments,
        num_nodes_for_multifocus: int,
    ) -> Tuple[e3nn.IrrepsArray, e3nn.SphericalSignal]:
        num_graphs, num_nodes_for_multifocus, num_targets, _ = graphs.globals.target_positions.shape
        num_nodes = graphs.nodes.positions.shape[0]

        # Compute the conditioning based on the focus nodes and target species.
        target_species = graphs.globals.target_species
        segment_ids = utils.get_segment_ids(graphs.n_node, num_nodes)

        def get_focus_indices_and_mask(segment_id: int):
            segment_mask = segment_ids == segment_id
            ndx = jnp.where(jnp.arange(1, num_nodes+1) * segment_mask * graphs.nodes.focus_mask, size=num_nodes_for_multifocus)[0]
            mask = ndx > 0
            return ndx - mask, mask
        focus_indices, focus_mask = jax.vmap(get_focus_indices_and_mask)(jnp.arange(num_graphs))
        conditioning = self.compute_conditioning(
            graphs, target_species, num_nodes_for_multifocus, focus_indices, focus_mask
        )

        target_positions = e3nn.IrrepsArray("1o", graphs.globals.target_positions)
        assert target_positions.shape == (
            num_graphs,
            num_nodes_for_multifocus,
            num_targets,
            3,
        ), target_positions.shape

        def predict_logits_for_single_graph(
            target_positions: e3nn.IrrepsArray, conditioning: e3nn.IrrepsArray
        ) -> Tuple[float, float]:
            """Predicts the logits for a single graph."""
            assert target_positions.shape == (num_targets, 3)
            assert conditioning.shape == (conditioning.irreps.dim,)

            radial_logits = hk.vmap(
                lambda pos: self.radial_predictor.log_prob(pos, conditioning),
                split_rng=False,
            )(target_positions)
            angular_logits = hk.vmap(
                lambda pos: self.angular_predictor.log_prob(pos, conditioning),
                split_rng=False,
            )(target_positions)
            return radial_logits, angular_logits

        radial_logits, angular_logits = hk.vmap(hk.vmap(
            predict_logits_for_single_graph, split_rng=False
        ), split_rng=False)(target_positions, conditioning)
        assert radial_logits.shape == (num_graphs, num_nodes_for_multifocus, num_targets)
        assert angular_logits.shape == (num_graphs, num_nodes_for_multifocus, num_targets)

        return radial_logits, angular_logits

    def get_evaluation_predictions(
        self,
        graphs: datatypes.Fragments,
        target_species: jnp.ndarray,
        inverse_temperature: float,
        num_nodes_for_multifocus: int,
        focus_indices: jnp.ndarray,
        focus_mask: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        num_graphs = graphs.n_node.shape[0]

        # Compute the conditioning based on the focus nodes and target species.
        conditioning = self.compute_conditioning(
            graphs,
            target_species,
            num_nodes_for_multifocus,
            focus_indices,
            focus_mask,)
        assert conditioning.shape == (num_graphs, num_nodes_for_multifocus, conditioning.irreps.dim)

        # Sample the radial component.
        radii = hk.vmap(hk.vmap(self.radial_predictor.sample, split_rng=True), split_rng=True)(conditioning)
        assert radii.shape == (num_graphs, num_nodes_for_multifocus)

        # Predict the target position vectors.
        angular_sample_fn = lambda r, cond: self.angular_predictor.sample(
            r, cond, inverse_temperature
        )
        position_vectors = hk.vmap(hk.vmap(angular_sample_fn, split_rng=True), split_rng=True)(
            radii, conditioning
        )
        assert position_vectors.shape == (num_graphs, num_nodes_for_multifocus, 3)
        return position_vectors
