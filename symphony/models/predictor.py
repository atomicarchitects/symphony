from typing import Union, Optional

import haiku as hk
import jax
import jax.numpy as jnp

from symphony import datatypes
from symphony.models.focus_predictor import FocusAndTargetSpeciesPredictor
from symphony.models.continuous_position_predictor import TargetPositionPredictor
from symphony.models.utils import utils


class Predictor(hk.Module):
    """A convenient wrapper for an entire prediction model."""

    def __init__(
        self,
        focus_and_target_species_predictor: FocusAndTargetSpeciesPredictor,
        target_position_predictor: TargetPositionPredictor,
        num_nodes_for_multifocus: int,
        num_targets: int,
        name: str = None,
    ):
        super().__init__(name=name)
        self.focus_and_target_species_predictor = focus_and_target_species_predictor
        self.target_position_predictor = target_position_predictor
        self.num_nodes_for_multifocus = num_nodes_for_multifocus
        self.num_targets = num_targets

    def get_training_predictions(
        self, graphs: datatypes.Fragments
    ) -> datatypes.Predictions:
        """Returns the predictions on these graphs during training, when we have access to the true focus and target species."""
        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]
        num_species = self.focus_and_target_species_predictor.num_species
        segment_ids = utils.get_segment_ids(graphs.n_node.reshape(-1,), num_nodes)

        # Get the species and stop logits.
        (
            focus_and_target_species_logits,
            stop_logits,
        ) = self.focus_and_target_species_predictor(graphs)

        # Get the species and stop probabilities.
        focus_and_target_species_probs, stop_probs = utils.segment_softmax_2D_with_stop(
            focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
        )

        # Get the focus node indices.
        # focus_node_indices = jnp.arange(num_nodes_for_multifocus)[graphs.nodes.focus_mask]

        # Get the logits at the target positions.
        (
            radial_logits,
            angular_logits,
        ) = self.target_position_predictor.get_training_predictions(
            graphs,
            self.num_nodes_for_multifocus
        )

        # Check the shapes.
        assert focus_and_target_species_logits.shape == (
            num_nodes,
            num_species,
        )
        assert focus_and_target_species_probs.shape == (
            num_nodes,
            num_species,
        )
        assert radial_logits.shape == (
            num_graphs,
            self.num_nodes_for_multifocus,
            self.num_targets,
        ), (radial_logits.shape, (num_graphs, self.num_nodes_for_multifocus, self.num_targets))
        assert angular_logits.shape == (
            num_graphs,
            self.num_nodes_for_multifocus,
            self.num_targets,
        )

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings_for_focus=self.focus_and_target_species_predictor.node_embedder(
                    graphs
                ),
                embeddings_for_positions=self.target_position_predictor.node_embedder(
                    graphs
                ),
                focus_mask=None,
                # target_species=None,
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                focus_indices=None,
                stop_logits=stop_logits,
                stop_probs=stop_probs,
                stop=None,
                target_species=None,
                radial_logits=radial_logits,
                angular_logits=angular_logits,
                position_vectors=None,
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
        """Returns the predictions on a single padded graph during evaluation, when we do not have access to the true focus and target species."""
        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]
        # num_nodes_for_multifocus = graphs.globals.target_positions.shape[1]
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
            focus_and_target_species_probs, segment_ids, num_graphs, focus_rng, self.num_nodes_for_multifocus
        )
        focus_mask = jnp.zeros((num_nodes,), dtype=jnp.bool_)
        focus_mask = focus_mask.at[focus_indices].set(True)

        # Compute the position coefficients.
        position_vectors = self.target_position_predictor.get_evaluation_predictions(
            graphs,
            target_species,
            position_inverse_temperature,
            self.num_nodes_for_multifocus,
            self.num_targets,
            focus_indices,
        )

        assert stop.shape == (num_graphs,)
        assert focus_indices.shape[0] == num_graphs
        assert focus_and_target_species_logits.shape == (
            num_nodes,
            num_species,
        )
        assert focus_and_target_species_probs.shape == (
            num_nodes,
            num_species,
        )
        assert len(position_vectors.shape) == 3
        assert position_vectors.shape[0] == num_graphs
        assert position_vectors.shape[2] == 3

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings_for_focus=self.focus_and_target_species_predictor.node_embedder(
                    graphs
                ),
                embeddings_for_positions=self.target_position_predictor.node_embedder(
                    graphs
                ),
                focus_mask=focus_mask,
                # target_species=target_species,
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop_logits=stop_logits,
                stop_probs=stop_probs,
                stop=stop,
                focus_indices=focus_indices,
                target_species=target_species,
                radial_logits=None,
                angular_logits=None,
                position_vectors=position_vectors,
            ),
            senders=graphs.senders,
            receivers=graphs.receivers,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )
