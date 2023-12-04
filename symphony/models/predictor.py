from typing import Union, Optional

import haiku as hk
import jax
import jax.numpy as jnp

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
        """Returns the predictions on graphs during training, when we have access to the true focus and target species."""

        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]
        num_species = self.focus_and_target_species_predictor.num_species
        segment_ids = utils.get_segment_ids(graphs.n_node, num_nodes)

        # Get the species and stop logits.
        (
            focus_and_target_species_logits,
            stop_logits,
        ) = self.focus_and_target_species_predictor(graphs)

        # Get the species and stop probabilities.
        focus_and_target_species_probs, stop_probs = utils.segment_softmax_2D_with_stop(
            focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
        )

        # Get the embeddings of the focus nodes.
        # These are the first nodes in each graph during training.
        focus_node_indices = utils.get_first_node_indices(graphs)

        # Get the coefficients for the target positions.
        (
            angular_logits,
            radial_logits,
        ) = self.target_position_predictor.predict_logits(
            graphs,
            focus_node_indices,
            graphs.globals.target_species,
            true_radii=jnp.linalg.norm(
                graphs.globals.target_positions,
                axis=-1,
            ),
            inverse_temperature=1.0,
        )

        # The radii bins used for the position prediction, repeated for each graph.
        radii = self.target_position_predictor.create_radii()
        radial_bins = jax.vmap(lambda _: radii)(jnp.arange(num_graphs))

        # Check the shapes.
        assert focus_and_target_species_logits.shape == (
            num_nodes,
            num_species,
        )
        assert focus_and_target_species_probs.shape == (
            num_nodes,
            num_species,
        )
        # assert log_position_coeffs.shape == (
        #     num_graphs,
        #     self.target_position_predictor.num_channels,
        #     self.target_position_predictor.num_radii,
        #     log_position_coeffs.shape[-1],
        # )
        # assert position_logits.shape == (
        #     num_graphs,
        #     self.target_position_predictor.num_radii,
        #     self.target_position_predictor.res_beta,
        #     self.target_position_predictor.res_alpha,
        # )

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings_for_focus=self.focus_and_target_species_predictor.compute_node_embeddings(
                    graphs
                ),
                embeddings_for_positions=self.target_position_predictor.compute_node_embeddings(
                    graphs
                ),
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop_logits=stop_logits,
                stop_probs=stop_probs,
                stop=None,
                focus_indices=focus_node_indices,
                target_species=None,
                log_position_coeffs=None,
                position_logits=None,
                position_probs=None,
                position_vectors=None,
                radial_bins=radial_bins,
                radial_logits=radial_logits,
                angular_logits=angular_logits,
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
    ) -> datatypes.Predictions:
        """Returns the predictions on graphs during evaluation, when we do not have access to the true focus and target species."""

        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]
        num_species = self.focus_and_target_species_predictor.num_species
        segment_ids = utils.get_segment_ids(graphs.n_node, num_nodes)

        # Get the species and stop logits.
        (
            focus_and_target_species_logits,
            stop_logits,
        ) = self.focus_and_target_species_predictor(
            graphs, inverse_temperature=focus_and_atom_type_inverse_temperature
        )

        # Get the softmaxed probabilities.
        focus_and_target_species_probs, stop_probs = utils.segment_softmax_2D_with_stop(
            focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
        )

        # Get the PRNG key for sampling.
        rng = hk.next_rng_key()

        # We stop a graph, if we sample a stop.
        rng, stop_rng = jax.random.split(rng)
        stop = jax.random.bernoulli(stop_rng, stop_probs)

        # Renormalize the focus and target species probabilities, if we have not stopped.
        focus_and_target_species_probs = focus_and_target_species_probs / (
            (1 - stop_probs)[segment_ids, None]
        )

        # Sample the focus node and target species.
        rng, focus_rng = jax.random.split(rng)
        focus_indices, target_species = utils.segment_sample_2D(
            focus_and_target_species_probs, segment_ids, num_graphs, focus_rng
        )

        # Sample the relative position vectors.
        position_vectors = self.target_position_predictor.sample(
            graphs,
            focus_indices,
            target_species,
            inverse_temperature=position_inverse_temperature,
        )

        assert stop.shape == (num_graphs,)
        assert focus_indices.shape == (num_graphs,)
        assert focus_and_target_species_logits.shape == (
            num_nodes,
            num_species,
        )
        assert focus_and_target_species_probs.shape == (
            num_nodes,
            num_species,
        )
        assert position_vectors.shape == (num_graphs, 3)

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings_for_focus=self.focus_and_target_species_predictor.compute_node_embeddings(
                    graphs
                ),
                embeddings_for_positions=self.target_position_predictor.compute_node_embeddings(
                    graphs
                ),
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop_logits=stop_logits,
                stop_probs=stop_probs,
                stop=stop,
                focus_indices=focus_indices,
                target_species=target_species,
                log_position_coeffs=None,
                position_logits=None,
                position_probs=None,
                position_vectors=position_vectors,
                radial_bins=None,
                radial_logits=None,
                angular_logits=None,
            ),
            senders=graphs.senders,
            receivers=graphs.receivers,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )
