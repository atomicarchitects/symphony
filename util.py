import e3nn_jax as e3nn
from e3nn_jax import to_s2grid, to_s2point
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
import jraph
from model import sample, DISTANCES, model_output


def loss_fn(key, graph, weights, mace_input, y, res_beta, res_alpha, quadrature, gamma=30):
    output = sample(key, weights, mace_input, res_beta, res_alpha, quadrature)
    
    return _loss(output, graph, y, res_beta, res_alpha, quadrature, gamma)


def _loss(output: model_output, graph: jraph.GraphTuple, y, res_beta, res_alpha, quadrature, gamma=30):
    """
    Args:
        output (ModelOutput):
        graph (jraph.GraphTuple): graph representing the current generated molecule
        y (GlobalsInfo): true global information about the target atom
    """
    loss_total = 0
    node_start = 0

    # TODO: is output also defined across multiple graphs?
    # TODO: is this good enough or do I need zip(graph.n_node, graph.n_edge)
    for n_nodes in graph.n_node:
        subgraph = subgraph(graph, range(node_start, node_start + n_nodes))
        node_start += n_nodes

        ## focus loss
        # assume that focus is the first element
        loss_focus = -1 * output.focus_logits[0] + logsumexp(output.focus_logits)
        focus_probs = jax.nn.softmax(output.focus_logits)
        correct_focus_prob = focus_probs[0]  # I feel like this is the way to go, we just need to get it to work

        ## atom type loss
        loss_type = -1 * output.atom_type_logits[y.target_atomic_number] + logsumexp(output.atom_type_logits)

        ## position loss
        position_signal = to_s2grid(output.position_coeffs, res_beta, res_alpha, quadrature=quadrature)
        pos_max = jnp.max(position_signal)
        prob_radius = position_signal.apply(lambda x: jnp.exp(x - pos_max)).integrate()

        # get radius weights for the "true" distribution
        target_pos = y.target_position - subgraph.nodes.positions[0]  # local position relative to focus
        radius_weights = jax.vmap(
            lambda dist_bucket: jnp.exp(-1 * (jnp.linalg.norm(target_pos) - dist_bucket)**2 / gamma),
            DISTANCES
        )
        radius_weights = radius_weights / jnp.sum(radius_weights)
        # getting f(r*, rhat*) [aka, our model's output for the true location]
        true_eval_pos = to_s2point(output.position_coeffs, target_pos)

        loss_pos = -jnp.sum(radius_weights * true_eval_pos) + jnp.log(jnp.sum(prob_radius)) + pos_max

        loss_total += loss_focus + (1 - output.stop) * (1 - correct_focus_prob) * (loss_type + loss_pos)

    ## return total loss
    return loss_total


def sample_on_s2grid(key, prob_s2, y, alpha, qw):
    """Sample points on the sphere

    Args:
        key (``jnp.ndarray``): random key
        prob_s2 (``jnp.ndarray``): shape ``(y, alpha)``, no need to be normalized
        y (``jnp.ndarray``): shape ``(y,)``
        alpha (``jnp.ndarray``): shape ``(alpha,)``
        qw (``jnp.ndarray``): shape ``(y,)``

    Returns:
        y_index (``jnp.ndarray``): shape ``()``
        alpha_index (``jnp.ndarray``): shape ``()``
    """
    k1, k2 = jax.random.split(key)
    p_ya = prob_s2  # [y, alpha]
    p_y = qw * jnp.sum(p_ya, axis=1)  # [y]
    y_index = jax.random.choice(k1, jnp.arange(len(y)), p=p_y)
    p_a = p_ya[y_index]  # [alpha]
    alpha_index = jax.random.choice(k2, jnp.arange(len(alpha)), p=p_a)
    return y_index, alpha_index


def subgraph(graph: jraph.GraphsTuple, nodes: jnp.ndarray) -> jraph.GraphsTuple:
    """Extract a subgraph from a graph.

    Args:
        graph: The graph to extract a subgraph from.
        nodes: The indices of the nodes to extract.

    Returns:
        The subgraph.
    """
    assert len(graph.n_edge) == 1 and len(graph.n_node) == 1, "Only single graphs supported."

    # Find all edges that connect to the nodes.
    edges = jnp.isin(graph.senders, nodes) & jnp.isin(graph.receivers, nodes)

    new_node_indices = -jnp.ones(graph.n_node[0], dtype=int)
    new_node_indices[nodes] = jnp.arange(len(nodes))

    return jraph.GraphsTuple(
        nodes=jax.tree_util.tree_map(lambda x: x[nodes], graph.nodes),
        edges=jax.tree_util.tree_map(lambda x: x[edges], graph.edges),
        globals=graph.globals,
        senders=new_node_indices[graph.senders[edges]],
        receivers=new_node_indices[graph.receivers[edges]],
        n_node=jnp.array([len(nodes)]),
        n_edge=jnp.array([jnp.sum(edges)]),
    )
