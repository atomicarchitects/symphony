import e3nn_jax as e3nn
from e3nn_jax import to_s2grid, to_s2point
import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
import jraph
from datatypes import GlobalsInfo, NodesInfo, ModelOutput


TYPES = ["H", "C", "N", "O", "F"]
DISTANCES = jnp.arange(0.05, 15.05, 0.05)


def _loss(
    output: ModelOutput,
    graph: jraph.GraphsTuple,
    res_beta,
    res_alpha,
    quadrature,
    gamma=30,
):
    """
    Args:
        output (ModelOutput):
        graph (jraph.GraphsTuple): graph representing the current generated molecule
        y (GlobalsInfo): true global information about the target atom
    """
    # indices for the true focus in each subgraph of the batch
    focus_true = jnp.concatenate([jnp.array([0]), jnp.cumsum(graph.n_node[:-1])])

    ## focus loss: node-based quantity
    # assume that focus is the first element
    loss_focus = -1 * output.focus_logits[focus_true] + logsumexp(output.focus_logits)
    focus_probs = jax.nn.softmax(output.focus_logits)
    correct_focus_probs = focus_probs[focus_true]

    ## atom type loss
    # TODO: this won't get jit'ed correctly?
    true_type_indices = jnp.vmap(lambda i, g: g.target_atomic_number + 5 * i)(
        enumerate(graph.globals)
    )
    loss_type = -1 * output.atom_type_logits[true_type_indices] + logsumexp(
        output.atom_type_logits
    )
    atom_type_dist = jax.nn.softmax(output.atom_type_logits)
    correct_type_probs = jax.vmap(lambda g: g.atom_type == jnp.argmax())(graph.globals)

    ## position loss
    position_signal = to_s2grid(
        output.position_coeffs, res_beta, res_alpha, quadrature=quadrature
    )
    pos_max = jnp.max(position_signal)
    prob_radius = position_signal.apply(lambda x: jnp.exp(x - pos_max)).integrate()

    # local position relative to focus
    target_pos = jax.vmap(lambda g: g.target_position)(graph.globals) - jnp.repeat(
        graph.nodes.positions[focus_true], graph.n_node, axis=0
    )
    # get radius weights for the "true" distribution
    radius_weights = jax.vmap(
        lambda dist_bucket: jnp.exp(
            -1 * (jnp.linalg.norm(target_pos) - dist_bucket) ** 2 / gamma
        )
    )(DISTANCES)
    radius_weights = radius_weights / jnp.sum(radius_weights)
    # getting f(r*, rhat*) [aka, our model's output for the true location]
    true_eval_pos = to_s2point(output.position_coeffs, target_pos)

    loss_pos = (
        -jnp.sum(radius_weights * true_eval_pos)
        + jnp.log(jnp.sum(prob_radius))
        + pos_max
    )

    ## return total loss
    losses = loss_focus + (1 - output.stop) * correct_focus_probs * (
        loss_type + correct_type_probs * loss_pos
    )
    return losses


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
