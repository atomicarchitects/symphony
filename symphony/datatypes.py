from typing import NamedTuple

import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph


class NodesInfo(NamedTuple):
    positions: jnp.ndarray  # [n_node, 3] float array
    species: jnp.ndarray  # [n_node] int array


class FragmentsGlobals(NamedTuple):
    stop: jnp.ndarray  # [n_graph] bool array (only for training)
    target_positions: jnp.ndarray  # [n_graph, num_nodes_for_multifocus, max_targets_per_graph, 3] float array (only for training)
    target_position_mask: jnp.ndarray  # [n_graph, num_nodes_for_multifocus, max_targets_per_graph] bool array (only for training)
    target_species: jnp.ndarray  # [n_graph, num_nodes_for_multifocus] int array (only for training)


class FragmentsNodes(NamedTuple):
    positions: jnp.ndarray  # [n_node, 3] float array
    species: jnp.ndarray  # [n_node] int array
    focus_and_target_species_probs: jnp.ndarray  # [n_node, n_species] float array (only for training)
    focus_mask: jnp.ndarray  # [n_node] bool array (only for training)

class Fragments(jraph.GraphsTuple):
    nodes: FragmentsNodes
    edges: None
    receivers: jnp.ndarray  # with integer dtype
    senders: jnp.ndarray  # with integer dtype
    globals: FragmentsGlobals
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray  # with integer dtype

    def from_graphstuple(graphs: jraph.GraphsTuple) -> "Fragments":
        return Fragments(
            nodes=graphs.nodes,
            edges=graphs.edges,
            receivers=graphs.receivers,
            senders=graphs.senders,
            globals=graphs.globals,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )


class NodePredictions(NamedTuple):
    embeddings_for_focus: e3nn.IrrepsArray  # [n_node, irreps] float array
    embeddings_for_positions: e3nn.IrrepsArray  # [n_node, irreps] float array
    focus_and_target_species_logits: jnp.ndarray  # [n_node, n_species] float array
    focus_and_target_species_probs: jnp.ndarray  # [n_node, n_species] float array
    focus_mask: jnp.ndarray  # [n_node] bool array
    target_species: jnp.ndarray  # [n_node,] int array


class GlobalPredictions(NamedTuple):
    focus_indices: jnp.ndarray  # [n_graph] int array
    stop_logits: jnp.ndarray  # [n_graph] float array
    stop_probs: jnp.ndarray  # [n_graph] float array
    stop: jnp.ndarray  # [n_graph] bool array
    position_logits: e3nn.SphericalSignal  # [num_nodes_for_multifocus, n_radii, beta, alpha] float array
    log_position_coeffs: e3nn.IrrepsArray  # [num_nodes_for_multifocus, n_radii, ...] float array
    position_probs: e3nn.SphericalSignal  # [num_nodes_for_multifocus, n_radii, beta, alpha] float array
    position_vectors: jnp.ndarray  # [num_nodes_for_multifocus, 3] float array
    radial_bins: jnp.ndarray  # [n_graph, n_radii] float array
    radial_logits: jnp.ndarray  # [num_nodes_for_multifocus, n_radii] float array
    angular_logits: e3nn.SphericalSignal  # [num_nodes_for_multifocus, n_radii, n_angular] float array


class Predictions(jraph.GraphsTuple):
    nodes: NodePredictions
    edges: None
    receivers: jnp.ndarray  # with integer dtype
    senders: jnp.ndarray  # with integer dtype
    globals: GlobalPredictions
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray  # with integer dtype
