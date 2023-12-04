from typing import NamedTuple

import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph


class NodesInfo(NamedTuple):
    positions: jnp.ndarray  # [n_node, 3] float array
    species: jnp.ndarray  # [n_node] int array


class FragmentsGlobals(NamedTuple):
    target_positions: jnp.ndarray  # [n_graph, 3] float array (only for training)
    target_species: jnp.ndarray  # [n_graph] int array (only for training)
    stop: jnp.ndarray  # [n_graph] bool array (only for training)


class FragmentsNodes(NamedTuple):
    positions: jnp.ndarray  # [n_node, 3] float array
    species: jnp.ndarray  # [n_node] int array
    focus_and_target_species_probs: jnp.ndarray  # [n_node, n_species] float array (only for training)


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


class GlobalPredictions(NamedTuple):
    stop_logits: jnp.ndarray  # [n_graph] float array
    stop_probs: jnp.ndarray  # [n_graph] float array
    stop: jnp.ndarray  # [n_graph] bool array
    focus_indices: jnp.ndarray  # [n_graph] int array
    target_species: jnp.ndarray  # [n_graph,] int array
    radial_logits: jnp.ndarray  # [n_graph, ] float array
    radii: jnp.ndarray  # [n_graph, ] float array
    radii_pdf: jnp.ndarray  # [n_graph, 1000] float array
    log_angular_coeffs: e3nn.IrrepsArray  # [n_graph, n_channels, ...] float array
    angular_logits: e3nn.SphericalSignal  # [n_graph, res_beta, res_alpha] float array
    angular_probs: e3nn.SphericalSignal  # [n_graph, res_beta, res_alpha] float array
    position_vectors: jnp.ndarray  # [n_graph, 3] float array


class Predictions(jraph.GraphsTuple):
    nodes: NodePredictions
    edges: None
    receivers: jnp.ndarray  # with integer dtype
    senders: jnp.ndarray  # with integer dtype
    globals: GlobalPredictions
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray  # with integer dtype
