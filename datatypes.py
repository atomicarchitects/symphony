from collections import namedtuple


GlobalsInfo = namedtuple("GlobalsInfo", ["stop", "target_position", "target_atomic_number"])
NodesInfo = namedtuple("NodesInfo", ["positions", "atomic_numbers"])

WeightTuple = namedtuple("WeightTuple", ["mace", "focus", "atom_type", "position"])
MaceInput = namedtuple("MACEinput", ["vectors", "atom_types", "senders", "receivers"])
ModelOutput = namedtuple("ModelOutput", [
    "stop", "focus_logits", "atom_type_logits", "position_coeffs"
])