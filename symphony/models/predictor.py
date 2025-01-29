from typing import Union, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph

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
            focus_and_target_species_logits,
            stop_logits,
            # big_logits,
        ) = self.focus_and_target_species_predictor(graphs)

        # Get the species and stop probabilities.
        focus_and_target_species_probs, stop_probs = utils.segment_softmax_2D_with_stop(
            focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
        )

        # Get the focus node indices.
        focus_node_indices = utils.get_first_node_indices(graphs)

        # segment_starts = jnp.concatenate(
        #     [jnp.zeros(1), jnp.cumsum(graphs.n_node)]
        # )
        # def f(i):
        #     mask1 = segment_starts[i] <= jnp.arange(num_nodes)
        #     mask2 = jnp.arange(num_nodes) < segment_starts[i + 1]
        #     mask = mask1 * mask2
        #     big_logits_masked = big_logits * mask[:, None]
        #     return e3nn.IrrepsArray(
        #         big_logits_masked.irreps,
        #         big_logits_masked.array.sum(axis=0) / jnp.maximum(mask.sum(), 1)
        #     )
        # big_logits = jax.vmap(f)(jnp.arange(num_graphs))

        # Get the logits at the target positions.
        (
            radial_logits,
            angular_logits,
        ) = self.target_position_predictor.get_training_predictions(
            graphs,
            # big_logits,
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
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop_logits=stop_logits,
                stop_probs=stop_probs,
                stop=None,
                focus_indices=focus_node_indices,
                target_species=graphs.globals.target_species,
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
        num_species = self.focus_and_target_species_predictor.num_species
        segment_ids = utils.get_segment_ids(graphs.n_node, num_nodes)

        # Get the species and stop logits.
        (
            focus_and_target_species_logits,
            stop_logits,
            # big_logits,
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

        # segment_starts = jnp.concatenate(
        #     [jnp.zeros(1), jnp.cumsum(graphs.n_node)]
        # )
        # def f(i):
        #     mask1 = segment_starts[i] <= jnp.arange(num_nodes)
        #     mask2 = jnp.arange(num_nodes) < segment_starts[i + 1]
        #     mask = mask1 * mask2
        #     big_logits_masked = big_logits * mask[:, None]
        #     return e3nn.IrrepsArray(
        #         big_logits_masked.irreps,
        #         big_logits_masked.array.sum(axis=0) / jnp.maximum(mask.sum(), 1)
        #     )
        # big_logits = jax.vmap(f)(jnp.arange(num_graphs))

        # Compute the position coefficients.
        radial_logits, angular_logits, position_vectors = self.target_position_predictor.get_evaluation_predictions(
            graphs,
            # big_logits,
            focus_indices,
            target_species,
            position_inverse_temperature,
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
                embeddings_for_focus=self.focus_and_target_species_predictor.node_embedder(
                    graphs
                ),
                embeddings_for_positions=self.target_position_predictor.node_embedder(
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
                radial_logits=radial_logits,
                angular_logits=angular_logits,
                position_vectors=position_vectors,
            ),
            senders=graphs.senders,
            receivers=graphs.receivers,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )
