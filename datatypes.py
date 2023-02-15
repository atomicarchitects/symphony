from collections import namedtuple


NodesInfo = namedtuple(
    "NodesInfo",
    [
        "positions",  # [n_node, 3] float array
        "atomic_numbers",  # [n_node] int array
    ],
)

TrainingGlobalsInfo = namedtuple(
    "TrainingGlobalsInfo",
    [
        "stop",  # [n_graph] bool array (only for training)
        "target_specie_probability",  # [n_graph, n_species] float array (only for training)
        "target_specie",  # [n_graph] int array (only for training)
        "target_position",  # [n_graph, 3] float array (only for training)
    ],
)
TrainingNodesInfo = namedtuple(
    "TrainingNodesInfo",
    [
        "positions",  # [n_node, 3] float array
        "species",  # [n_node] int array
        "focus_probability",  # [n_node] float array (only for training)
    ],
)

WeightTuple = namedtuple("WeightTuple", ["mace", "focus", "atom_type", "position"])
MaceInput = namedtuple("MACEinput", ["vectors", "atom_types", "senders", "receivers"])
ModelOutput = namedtuple(
    "ModelOutput", ["stop", "focus_logits", "atom_type_logits", "position_coeffs"]
)
