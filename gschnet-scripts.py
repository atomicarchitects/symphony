import jax
import jax.numpy as jnp

import analysis
import datatypes
from input_pipeline_tf import get_datasets


config, best_state, metrics_for_best_state = analysis.load_from_workdir('workdirs/mace0')

cutoff = 5.0
key = jax.random.PRNGKey(0)
epsilon = 1e-4

qm9_datasets = get_datasets(key, config)
example_graph = next(qm9_datasets["test"].as_numpy_iterator())
frag = datatypes.Fragment.from_graphstuple(example_graph)
frag = jax.tree_map(jnp.asarray, frag)

generated_dict = analysis.to_mol_dict(frag, "workdirs", "generated_single_frag")