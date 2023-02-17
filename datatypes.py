from collections import namedtuple

import jax.numpy as jnp
import jraph

NodesInfo = namedtuple(
    "NodesInfo",
    [
        "positions",  # [n_node, 3] float array
        "species",  # [n_node] int array
    ],
)

FragmentGlobals = namedtuple(
    "FragmentGlobals",
    [
        "stop",  # [n_graph] bool array (only for training)
        "target_positions",  # [n_graph, 3] float array (only for training)
        "target_species",  # [n_graph] int array (only for training)
        "target_species_probability",  # [n_graph, n_species] float array (only for training)
    ],
)

FragmentNodes = namedtuple(
    "FragmentNodes",
    [
        "positions",  # [n_node, 3] float array
        "species",  # [n_node] int array
        "focus_probability",  # [n_node] float array (only for training)
    ],
)


class Fragment(jraph.GraphsTuple):
    nodes: FragmentNodes
    edges: None
    receivers: jnp.ndarray  # with integer dtype
    senders: jnp.ndarray  # with integer dtype
    globals: FragmentGlobals
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray  # with integer dtype


WeightTuple = namedtuple("WeightTuple", ["mace", "focus", "atom_type", "position"])
MaceInput = namedtuple("MACEinput", ["vectors", "atom_types", "senders", "receivers"])
Predictions = namedtuple(
    "Predictions", ["focus_logits", "species_logits", "position_coeffs"]
)
