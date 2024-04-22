from typing import Tuple

import jax
import jax.numpy as jnp
import jraph

from symphony import datatypes, models


@jax.profiler.annotate_function
def generation_loss(
    preds: datatypes.Predictions,
    graphs: datatypes.Fragments,
    ignore_position_loss_for_small_fragments: bool,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Computes the loss for the generation task.
    Args:
    - preds: the model predictions
    - graphs: a batch of graphs representing the current molecules
    - ignore_position_loss_for_small_fragments: whether to ignore the position loss for fragments with less than or equal to 3 atoms
    """
    num_targets = graphs.globals.target_positions_mask.shape[-1]
    num_graphs = graphs.n_node.shape[0]
    num_nodes = graphs.nodes.positions.shape[0]
    n_node = graphs.n_node
    segment_ids = models.get_segment_ids(n_node, num_nodes)

    def focus_and_atom_type_loss() -> jnp.ndarray:
        """Computes the loss over focus and atom types for all nodes."""
        species_logits = preds.nodes.focus_and_target_species_logits
        species_targets = graphs.nodes.focus_and_target_species_probs
        stop_logits = preds.globals.stop_logits
        stop_targets = graphs.globals.stop.astype(jnp.float32)

        assert species_logits.shape == (num_nodes, species_logits.shape[-1])
        assert species_targets.shape == (num_nodes, species_logits.shape[-1])
        assert stop_logits.shape == (num_graphs,)
        assert stop_targets.shape == (num_graphs,)

        # Subtract the maximum value for numerical stability.
        # This doesn't affect the forward pass, nor the backward pass.
        logits_max = jraph.segment_max(
            species_logits, segment_ids, num_segments=num_graphs
        ).max(axis=-1)
        logits_max = jnp.maximum(logits_max, stop_logits)
        logits_max = jax.lax.stop_gradient(logits_max)
        species_logits -= logits_max[segment_ids, None]
        stop_logits -= logits_max

        # Compute the cross-entropy loss.
        loss_focus_and_atom_type = -(species_targets * species_logits).sum(axis=-1)
        loss_focus_and_atom_type = jraph.segment_sum(
            loss_focus_and_atom_type, segment_ids, num_graphs
        )
        loss_focus_and_atom_type += -stop_targets * stop_logits
        loss_focus_and_atom_type += jnp.log(
            jraph.segment_sum(
                jnp.exp(species_logits).sum(axis=-1), segment_ids, num_graphs
            )
            + jnp.exp(stop_logits)
        )

        # Compute the lower bound on cross-entropy loss as the entropy of the target distribution.
        lower_bounds = -(species_targets * models.safe_log(species_targets)).sum(
            axis=-1
        )
        lower_bounds = jraph.segment_sum(lower_bounds, segment_ids, num_graphs)
        lower_bounds += -stop_targets * models.safe_log(stop_targets)
        lower_bounds = jax.lax.stop_gradient(lower_bounds)

        # Subtract out self-entropy (lower bound) to get the KL divergence.
        loss_focus_and_atom_type -= lower_bounds
        assert loss_focus_and_atom_type.shape == (num_graphs,)

        return loss_focus_and_atom_type

    def position_loss() -> jnp.ndarray:
        """Computes the loss over position probabilities."""
        assert graphs.globals.target_positions.shape == (num_graphs, num_targets, 3), (
            graphs.globals.target_positions.shape,
            num_graphs,
            num_targets,
            3,
        )
        position_logits = preds.globals.radial_logits + preds.globals.angular_logits
        assert position_logits.shape == (num_graphs, num_targets), (
            position_logits.shape,
            num_graphs,
            num_targets,
        )

        target_positions_mask = graphs.globals.target_positions_mask
        assert target_positions_mask.shape == (num_graphs, num_targets)

        loss_position = -position_logits
        loss_position = jnp.where(target_positions_mask, loss_position, 0)
        loss_position = loss_position.sum(axis=-1)
        num_valid_targets = jnp.maximum(1, target_positions_mask.sum(axis=-1))
        loss_position /= num_valid_targets

        # jax.debug.print("target_positions={x}", x=graphs.globals.target_positions)
        # jax.debug.print("radial_logits={x}", x=preds.globals.radial_logits)
        # jax.debug.print("angular_logits={x}", x=preds.globals.angular_logits)
        # jax.debug.print("mask={x}", x=graphs.globals.target_positions_mask)
        # jax.debug.print("loss_position={x}", x=loss_position)
        # jax.debug.print("")

        assert loss_position.shape == (num_graphs,)

        return loss_position

    # If we should predict a STOP for this fragment, we do not have to predict a position.
    loss_focus_and_atom_type = focus_and_atom_type_loss()
    loss_position = (1 - graphs.globals.stop) * position_loss()

    # Ignore position loss for graphs with less than, or equal to 3 atoms?
    # This is because there are symmetry-based degeneracies in the target distribution for these graphs.
    if ignore_position_loss_for_small_fragments:
        loss_position = jnp.where(n_node <= 3, 0, loss_position)

    total_loss = loss_focus_and_atom_type + loss_position
    return total_loss, (loss_focus_and_atom_type, loss_position)
