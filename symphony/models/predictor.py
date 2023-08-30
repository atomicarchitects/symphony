from typing import Union, Optional

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp

from symphony import datatypes
from symphony.models.focus_predictor import FocusAndTargetSpeciesPredictor
from symphony.models.position_predictor import FactorizedTargetPositionPredictor, TargetPositionPredictor
from symphony.models.embedders.global_embedder import GlobalEmbedder
from symphony.models import utils


class Predictor(hk.Module):
    """A convenient wrapper for an entire prediction model."""

    def __init__(
        self,
        node_embedder_for_focus: hk.Module,
        node_embedder_for_positions: hk.Module,
        focus_and_target_species_predictor: FocusAndTargetSpeciesPredictor,
        target_position_predictor: Union[
            TargetPositionPredictor, FactorizedTargetPositionPredictor
        ],
        global_embedder: Optional[GlobalEmbedder] = None,
        name: str = None,
    ):
        super().__init__(name=name)
        self.node_embedder_for_focus = node_embedder_for_focus
        self.node_embedder_for_positions = node_embedder_for_positions
        self.global_embedder = global_embedder
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

        # Get the node embeddings.
        node_embeddings = self.node_embedder_for_focus(graphs)

        # Concatenate global embeddings to node embeddings.
        if self.global_embedder is not None:
            graphs_with_node_embeddings = graphs._replace(nodes=node_embeddings)
            global_embeddings = self.global_embedder(graphs_with_node_embeddings)
            node_embeddings = e3nn.concatenate(
                [node_embeddings, global_embeddings], axis=-1
            )

        # Get the species and stop logits.
        focus_and_target_species_logits = self.focus_and_target_species_predictor(
            node_embeddings
        )
        stop_logits = jnp.zeros((num_graphs,))

        # Get the species and stop probabilities.
        focus_and_target_species_probs, stop_probs = utils.segment_softmax_2D_with_stop(
            focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
        )

        # Get the embeddings of the focus nodes.
        # These are the first nodes in each graph during training.
        auxiliary_node_embeddings = self.node_embedder_for_positions(graphs)
        focus_node_indices = utils.get_first_node_indices(graphs)
        true_focus_node_embeddings = auxiliary_node_embeddings[focus_node_indices]

        # Get the position coefficients.
        res_beta, res_alpha, num_radii = (
            self.target_position_predictor.res_beta,
            self.target_position_predictor.res_alpha,
            self.target_position_predictor.num_radii,
        )
        if isinstance(self.target_position_predictor, TargetPositionPredictor):
            angular_logits, radial_logits = None, None
            position_coeffs = self.target_position_predictor(
                true_focus_node_embeddings, graphs.globals.target_species
            )
        elif isinstance(
            self.target_position_predictor, FactorizedTargetPositionPredictor
        ):
            radial_logits, log_angular_coeffs = self.target_position_predictor(
                true_focus_node_embeddings, graphs.globals.target_species
            )
            angular_logits = jax.vmap(
                lambda coeffs: utils.log_coeffs_to_logits(coeffs, res_beta, res_alpha, 1)
            )(
                log_angular_coeffs[:, None, :]
            )  # only one radius
            # Mix the radial components with each channel of the angular components.
            position_coeffs = jax.vmap(
                jax.vmap(
                    utils.compute_coefficients_of_logits_of_joint_distribution,
                    in_axes=(None, 0),
                )
            )(radial_logits, log_angular_coeffs)

        # Compute the position signal projected to a spherical grid for each radius.
        position_logits = jax.vmap(
            lambda coeffs: utils.log_coeffs_to_logits(coeffs, res_beta, res_alpha, num_radii)
        )(position_coeffs)
        assert position_logits.shape == (
            num_graphs,
            num_radii,
            res_beta,
            res_alpha,
        )

        # Get the position probabilities.
        position_probs = jax.vmap(utils.position_logits_to_position_distribution)(
            position_logits
        )

        # The radii bins used for the position prediction, repeated for each graph.
        radii = self.target_position_predictor.create_radii()
        radial_bins = jnp.tile(radii, (num_graphs, 1))

        # Check the shapes.
        assert focus_and_target_species_logits.shape == (
            num_nodes,
            num_species,
        ), focus_and_target_species_logits.shape
        assert focus_and_target_species_probs.shape == (
            num_nodes,
            num_species,
        ), focus_and_target_species_probs.shape
        assert position_coeffs.shape == (
            num_graphs,
            self.target_position_predictor.num_channels,
            self.target_position_predictor.num_radii,
            position_coeffs.shape[-1],
        ), position_coeffs.shape
        assert position_logits.shape == (
            num_graphs,
            self.target_position_predictor.num_radii,
            self.target_position_predictor.res_beta,
            self.target_position_predictor.res_alpha,
        )

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings=node_embeddings,
                auxiliary_node_embeddings=auxiliary_node_embeddings,
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop_logits=stop_logits,
                stop_probs=stop_probs,
                stop=None,
                focus_indices=focus_node_indices,
                target_species=None,
                position_coeffs=position_coeffs,
                position_logits=position_logits,
                position_probs=position_probs,
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
        """Returns the predictions on a single padded graph during evaluation, when we do not have access to the true focus and target species."""
        # Get the number of graphs and nodes.
        num_nodes = graphs.nodes.positions.shape[0]
        num_graphs = graphs.n_node.shape[0]
        num_species = self.focus_and_target_species_predictor.num_species
        segment_ids = utils.get_segment_ids(graphs.n_node, num_nodes)

        # Get the PRNG key for sampling.
        rng = hk.next_rng_key()

        # Get the node embeddings.
        node_embeddings_for_focus = self.node_embedder_for_focus(graphs)

        # Concatenate global embeddings to node embeddings.
        if self.global_embedder is not None:
            graphs_with_node_embeddings = graphs._replace(nodes=node_embeddings_for_focus)
            global_embeddings = self.global_embedder(graphs_with_node_embeddings)
            node_embeddings = e3nn.concatenate(
                [node_embeddings, global_embeddings], axis=-1
            )

        # Get the species and stop logits.
        focus_and_target_species_logits = self.focus_and_target_species_predictor(
            node_embeddings
        )
        stop_logits = jnp.zeros((num_graphs,))

        # Scale the logits by the inverse temperature.
        focus_and_target_species_logits *= focus_and_atom_type_inverse_temperature
        stop_logits *= focus_and_atom_type_inverse_temperature

        # Get the softmaxed probabilities.
        focus_and_target_species_probs, stop_probs = utils.segment_softmax_2D_with_stop(
            focus_and_target_species_logits, stop_logits, segment_ids, num_graphs
        )

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

        # Get the embeddings of the focus node.
        auxiliary_node_embeddings = self.node_embedder_for_positions(graphs)
        focus_node_embeddings = auxiliary_node_embeddings[focus_indices]

        # Get the position coefficients.
        res_beta, res_alpha, num_radii = (
            self.target_position_predictor.res_beta,
            self.target_position_predictor.res_alpha,
            self.target_position_predictor.num_radii,
        )
        if isinstance(self.target_position_predictor, TargetPositionPredictor):
            angular_logits, radial_logits = None, None
            position_coeffs = self.target_position_predictor(
                focus_node_embeddings, target_species
            )
        elif isinstance(
            self.target_position_predictor, FactorizedTargetPositionPredictor
        ):
            radial_logits, log_angular_coeffs = self.target_position_predictor(
                focus_node_embeddings, target_species
            )
            angular_logits = jax.vmap(
                lambda coeffs: utils.log_coeffs_to_logits(coeffs, res_beta, res_alpha, 1)
            )(log_angular_coeffs[:, None, :])
            # Mix the radial components with each channel of the angular components.
            position_coeffs = jax.vmap(
                jax.vmap(
                    utils.compute_coefficients_of_logits_of_joint_distribution,
                    in_axes=(None, 0),
                )
            )(radial_logits, log_angular_coeffs)

        # Scale by inverse temperature.
        position_coeffs = position_inverse_temperature * position_coeffs

        # Compute the position signal projected to a spherical grid for each radius.
        position_logits = jax.vmap(
            lambda coeffs: utils.log_coeffs_to_logits(coeffs, res_beta, res_alpha, num_radii)
        )(position_coeffs)

        # Integrate the position signal over each sphere to get the normalizing factors for the radii.
        # For numerical stability, we subtract out the maximum value over all spheres before exponentiating.
        position_probs = jax.vmap(utils.position_logits_to_position_distribution)(
            position_logits
        )

        # Sample the radius.
        radii = self.target_position_predictor.create_radii()
        radial_bins = jnp.tile(radii, (num_graphs, 1))
        radial_probs = jax.vmap(utils.position_distribution_to_radial_distribution)(
            position_probs
        )
        num_radii = radii.shape[0]
        rng, radius_rng = jax.random.split(rng)
        radius_rngs = jax.random.split(radius_rng, num_graphs)
        radius_indices = jax.vmap(
            lambda key, p: jax.random.choice(key, num_radii, p=p)
        )(
            radius_rngs, radial_probs
        )  # [num_graphs]

        # Get the angular probabilities.
        angular_probs = jax.vmap(
            lambda p, r_index: p[r_index] / p[r_index].integrate()
        )(
            position_probs, radius_indices
        )  # [num_graphs, res_beta, res_alpha]

        # Sample angles.
        rng, angular_rng = jax.random.split(rng)
        angular_rngs = jax.random.split(angular_rng, num_graphs)
        beta_indices, alpha_indices = jax.vmap(lambda key, p: p.sample(key))(
            angular_rngs, angular_probs
        )

        # Combine the radius and angles to get the position vectors.
        position_vectors = jax.vmap(
            lambda r, b, a: radii[r] * angular_probs.grid_vectors[b, a]
        )(radius_indices, beta_indices, alpha_indices)

        # Check the shapes.
        irreps = e3nn.s2_irreps(self.target_position_predictor.position_coeffs_lmax)

        assert stop.shape == (num_graphs,)
        assert focus_indices.shape == (num_graphs,)
        assert focus_and_target_species_logits.shape == (num_nodes, num_species)
        assert focus_and_target_species_probs.shape == (num_nodes, num_species)
        assert position_coeffs.shape == (
            num_graphs,
            self.target_position_predictor.num_channels,
            num_radii,
            irreps.dim,
        )
        assert position_logits.shape == (
            num_graphs,
            self.target_position_predictor.num_radii,
            res_beta,
            res_alpha,
        )
        assert position_vectors.shape == (num_graphs, 3)

        return datatypes.Predictions(
            nodes=datatypes.NodePredictions(
                focus_and_target_species_logits=focus_and_target_species_logits,
                focus_and_target_species_probs=focus_and_target_species_probs,
                embeddings=node_embeddings,
                auxiliary_node_embeddings=auxiliary_node_embeddings,
            ),
            edges=None,
            globals=datatypes.GlobalPredictions(
                stop_logits=stop_logits,
                stop_probs=stop_probs,
                stop=stop,
                focus_indices=focus_indices,
                target_species=target_species,
                position_coeffs=position_coeffs,
                position_logits=position_logits,
                position_probs=position_probs,
                position_vectors=position_vectors,
                radial_bins=radial_bins,
                radial_logits=radial_logits,
                angular_logits=angular_logits,
            ),
            senders=graphs.senders,
            receivers=graphs.receivers,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )


