from typing import NamedTuple, Optional

import e3nn_jax as e3nn
import jax.numpy as jnp
import jraph


class NodesInfo(NamedTuple):
    positions: jnp.ndarray  # [n_node, 3] float array
    species: jnp.ndarray  # [n_node] int array


class FragmentGlobals(NamedTuple):
    stop: jnp.ndarray  # [n_graph] bool array (only for training)
    target_positions: jnp.ndarray  # [n_graph, 3] float array (only for training)
    target_species: jnp.ndarray  # [n_graph] int array (only for training)
    target_species_probability: jnp.ndarray  # [n_graph, n_species] float array (only for training)


class FragmentNodes(NamedTuple):
    positions: jnp.ndarray  # [n_node, 3] float array
    species: jnp.ndarray  # [n_node] int array
    focus_probability: jnp.ndarray  # [n_node] float array (only for training)


class Fragment(jraph.GraphsTuple):
    nodes: FragmentNodes
    edges: Optional[jnp.ndarray]
    receivers: jnp.ndarray  # with integer dtype
    senders: jnp.ndarray  # with integer dtype
    globals: FragmentGlobals
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray  # with integer dtype

    def from_graphstuple(graphs: jraph.GraphsTuple) -> "Fragment":
        return Fragment(
            nodes=graphs.nodes,
            edges=graphs.edges,
            receivers=graphs.receivers,
            senders=graphs.senders,
            globals=graphs.globals,
            n_node=graphs.n_node,
            n_edge=graphs.n_edge,
        )


class PredictionsNodes(NamedTuple):
    focus_logits: jnp.ndarray  # [n_node] float array


class PredictionsGlobals(NamedTuple):
    focus_indices: jnp.ndarray  # [n_graph] int array
    stop: jnp.ndarray  # [n_graph] bool array

    target_species_logits: jnp.ndarray  # [n_graph, n_species] float array
    target_species: jnp.ndarray  # [n_graph,] int array

    position_coeffs: jnp.ndarray  # [n_graph, n_radii, ...] float array
    position_logits: e3nn.SphericalSignal  # [n_graph, n_radii, beta, alpha] float array
    position_probs: e3nn.SphericalSignal  # [n_graph, n_radii, beta, alpha] float array
    position_vectors: jnp.ndarray  # [n_graph, 3] float array
