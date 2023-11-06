from typing import Union, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import jraph

from symphony import datatypes
from symphony.models.focus_predictor import FocusAndTargetSpeciesPredictor
from symphony.models.position_predictor import (
    FactorizedTargetPositionPredictor,
    TargetPositionPredictor,
)
from symphony.models import utils


class Predictor(hk.Module):
    """A convenient wrapper for an entire prediction model."""

    def __init__(
        self,
        focus_and_target_species_predictor: FocusAndTargetSpeciesPredictor,
        target_position_predictor: Union[
            TargetPositionPredictor, FactorizedTargetPositionPredictor
        ],
        name: str = None,
    ):
        super().__init__(name=name)
        self.focus_and_target_species_predictor = focus_and_target_species_predictor
        self.target_position_predictor = target_position_predictor

    def get_training_predictions(
        self, graphs: datatypes.Fragments
    ) -> datatypes.Predictions:
        """Returns the predictions on these graphs during training, when we have access to the true focus and target species."""
        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]
        num_species = self.focus_and_target_species_predictor.num_species
        segment_ids = utils.get_segment_ids(graphs.n_node, num_nodes)

        # Get the species and stop logits.
        (
            focus_logits,
            target_species_logits,
        ) = self.focus_and_target_species_predictor(graphs)

        # Get the species and stop probabilities.
        focus_probs = jax.nn.sigmoid(focus_logits)
        target_species_probs = jax.vmap(jax.nn.softmax)(target_species_logits)

        # Get the coefficients for the target positions.
        (
            log_position_coeffs,
            position_logits,
            angular_logits,
            radial_logits,
        ) = self.target_position_predictor(
            graphs,
            graphs.nodes.target_species,
            inverse_temperature=1.0,
        )

        # Get the position probabilities.
        position_probs = jax.vmap(utils.position_logits_to_position_distribution)(
            position_logits
        )

        # The radii bins used for the position prediction, repeated for each graph.
        radii = self.target_position_predictor.create_radii()
        radial_bins = jax.vmap(lambda _: radii)(jnp.arange(num_nodes))

        # Check the shapes.
        assert focus_logits.shape == (num_nodes,)
        assert target_species_logits.shape == (
            num_nodes,
            num_species,
        )
        assert log_position_coeffs.shape == (
            num_nodes,
            self.target_position_predictor.num_channels,
            self.target_position_predictor.num_radii,
            log_position_coeffs.shape[-1],
        )
        assert position_logits.shape == (
            num_nodes,
            self.target_position_predictor.num_radii,
            self.target_position_predictor.res_beta,
            self.target_position_predictor.res_alpha,
        )

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                embeddings_for_focus=self.focus_and_target_species_predictor.compute_node_embeddings(
                    graphs
                ),
                embeddings_for_positions=self.target_position_predictor.compute_node_embeddings(
                    graphs
                ),
                focus_logits=focus_logits,
                focus_probs=focus_probs,
                focus_mask=None,
                target_species_logits=target_species_logits,
                target_species_probs=target_species_probs,
                target_species=None,
                log_position_coeffs=log_position_coeffs,
                position_logits=position_logits,
                position_probs=position_probs,
                position_vectors=None,
                radial_bins=radial_bins,
                radial_logits=radial_logits,
                angular_logits=angular_logits,
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop=None,
            ),
            senders=graphs.senders,
            receivers=graphs.receivers,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )

    def get_evaluation_predictions(
        self,
        graphs: datatypes.Fragments,
        focus_and_atom_type_inverse_temperature: float,
        position_inverse_temperature: float,
        num_nodes_for_multifocus: int = 5,
    ) -> datatypes.Predictions:
        """Returns the predictions on a single padded graph during evaluation, when we do not have access to the true focus and target species."""
        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]
        num_species = self.focus_and_target_species_predictor.num_species
        segment_ids = utils.get_segment_ids(graphs.n_node, num_nodes)

        # Get the species and stop logits.
        (
            focus_logits,
            target_species_logits,
        ) = self.focus_and_target_species_predictor(
            graphs, inverse_temperature=focus_and_atom_type_inverse_temperature
        )

        # Get the softmaxed probabilities.
        focus_probs = jax.nn.sigmoid(focus_logits)
        target_species_probs = jax.vmap(jax.nn.softmax)(target_species_logits)

        # Get the PRNG key for sampling.
        rng = hk.next_rng_key()


        # Sample the focus node and target species.
        rng, focus_rng = jax.random.split(rng)
        focus_mask = jax.random.bernoulli(focus_rng, focus_probs)

        # If we have less than 'num_nodes_for_multifocus' nodes, we only choose one focus.
        # We choose the focus with the highest probability.
        def focus_mask_for_graph(index):
            focus_probs_for_graph = jnp.where(segment_ids == index, focus_probs, 0.)
            max_index = jnp.argmax(focus_probs_for_graph)
            has_enough_nodes = (graphs.n_node[index] >= num_nodes_for_multifocus)
            focus_mask_for_graph = jnp.where(segment_ids == index, jnp.logical_or(jnp.arange(num_nodes) == max_index, has_enough_nodes), 1.)
            return focus_mask_for_graph
        focus_mask *= jnp.all(jax.vmap(focus_mask_for_graph)(jnp.arange(num_graphs)), axis=0)

        rng, target_species_rng = jax.random.split(rng)
        target_species_rngs = jax.random.split(target_species_rng, num_nodes)
        target_species = jax.vmap(jax.random.categorical)(
            target_species_rngs, target_species_logits
        )

        # Compute the position coefficients.
        (
            log_position_coeffs,
            position_logits,
            angular_logits,
            radial_logits,
        ) = self.target_position_predictor(
            graphs,
            target_species,
            inverse_temperature=position_inverse_temperature,
        )

        # Integrate the position signal over each sphere to get the normalizing factors for the radii.
        # For numerical stability, we subtract out the maximum value over all spheres before exponentiating.
        position_probs = jax.vmap(utils.position_logits_to_position_distribution)(
            position_logits
        )

        # Sample the radius.
        radii = self.target_position_predictor.create_radii()
        rng, position_rng = jax.random.split(rng)
        position_rngs = jax.random.split(position_rng, num_nodes)
        position_vectors = jax.vmap(
            utils.sample_from_position_distribution, in_axes=(0, None, 0)
        )(position_probs, radii, position_rngs)

        # The radii bins used for the position prediction, repeated for each node.
        radial_bins = jax.vmap(lambda _: radii)(jnp.arange(num_nodes))

        # We stop a graph, if none of the nodes were selected as the focus.
        stop = jraph.segment_sum(focus_mask.astype(jnp.float32), segment_ids, num_graphs) == 0

        assert stop.shape == (num_graphs,)
        assert focus_logits.shape == (num_nodes,)
        assert target_species_logits.shape == (
            num_nodes,
            num_species,
        )
        assert log_position_coeffs.shape == (
            num_nodes,
            self.target_position_predictor.num_channels,
            self.target_position_predictor.num_radii,
            log_position_coeffs.shape[-1],
        )
        assert position_logits.shape == (
            num_nodes,
            self.target_position_predictor.num_radii,
            self.target_position_predictor.res_beta,
            self.target_position_predictor.res_alpha,
        )
        assert position_vectors.shape == (num_nodes, 3)

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                embeddings_for_focus=self.focus_and_target_species_predictor.compute_node_embeddings(
                    graphs
                ),
                embeddings_for_positions=self.target_position_predictor.compute_node_embeddings(
                    graphs
                ),
                focus_logits=focus_logits,
                focus_probs=focus_probs,
                target_species_logits=target_species_logits,
                target_species_probs=target_species_probs,
                focus_mask=focus_mask,
                target_species=target_species,
                log_position_coeffs=log_position_coeffs,
                position_logits=position_logits,
                position_probs=position_probs,
                position_vectors=position_vectors,
                radial_bins=radial_bins,
                radial_logits=radial_logits,
                angular_logits=angular_logits,
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop=stop,
            ),
            senders=graphs.senders,
            receivers=graphs.receivers,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )
