"""Generates molecules from a trained model."""

from typing import Sequence, Tuple

import os
import sys

from absl import flags
from absl import app
from absl import logging
import ase
import ase.data
from ase.db import connect
import ase.io
import ase.visualize
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import tqdm
import chex
import matscipy.neighbours
from plotly import graph_objects as go
from plotly.subplots import make_subplots

sys.path.append("..")

import analyses.analysis as analysis
import datatypes  # noqa: E402
import input_pipeline  # noqa: E402
import models  # noqa: E402

MAX_NUM_ATOMS = 30

FLAGS = flags.FLAGS


def get_edge_padding_mask(
    n_node: jnp.ndarray, n_edge: jnp.ndarray, sum_n_edge: int
) -> jnp.ndarray:
    return jraph.get_edge_padding_mask(
        jraph.GraphsTuple(
            nodes=None,
            edges=None,
            globals=None,
            receivers=None,
            senders=jnp.zeros(sum_n_edge, dtype=jnp.int32),
            n_node=n_node,
            n_edge=n_edge,
        )
    )


def create_radius_graph(
    positions: jnp.ndarray, n_node: jnp.ndarray, cutoff: float, sum_n_edge: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create a radius graph.

    Args:
        positions (jnp.ndarray): (n_node_sum, 3) array of positions, assumed to be padded, see `jraph.pad_with_graphs`
        n_node (jnp.ndarray): (n_graph,) array of number of nodes per graph, see `jraph.batch_np` and `jraph.pad_with_graphs`
        cutoff (float): cutoff radius
        sum_n_edge (int): the total number of edges in output graph, the output graph will be padded with `jraph.pad_with_graphs`

    Returns:
        senders: jnp.ndarray of dimension (sum_n_edge,) array of sender indices
        receivers: jnp.ndarray of dimension (sum_n_edge,) array of receiver indices
        n_edge: jnp.ndarray of dimension (n_graph,) array of number of edges per graph
        bool: True if the radius graph was created successfully
    """
    return jax.pure_callback(
        _create_radius_graph_helper,
        (
            jnp.empty(sum_n_edge, dtype=jnp.int32),  # senders
            jnp.empty(sum_n_edge, dtype=jnp.int32),  # receivers
            jnp.empty(n_node.shape, dtype=jnp.int32),  # n_edge
            jnp.empty((), dtype=jnp.bool_),  # ok
        ),
        jax.lax.stop_gradient(positions),
        n_node,
        cutoff,
        sum_n_edge,
    )


def _create_radius_graph_helper(
    positions: np.ndarray, n_node: np.ndarray, cutoff: float, sum_n_edge: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a radius graph.

    Args:
        positions (jnp.ndarray): (n_node_sum, 3) array of positions, assumed to be padded, see `jraph.pad_with_graphs`
        n_node (jnp.ndarray): (n_graph,) array of number of nodes per graph, see `jraph.batch_np` and `jraph.pad_with_graphs`
        cutoff (float): cutoff radius
        sum_n_edge (int): the total number of edges in output graph, the output graph will be padded with `jraph.pad_with_graphs`

    Returns:
        jnp.ndarray: senders: (sum_n_edge,) array of sender indices
        jnp.ndarray: receivers: (sum_n_edge,) array of receiver indices
        jnp.ndarray: n_edge: (n_graph,) array of number of edges per graph
        bool: True if the radius graph was created successfully
    """
    sum_n_node = positions.shape[0]
    n_graph = n_node.shape[0]

    graph = jraph.GraphsTuple(
        nodes=positions.copy(),
        edges=None,
        globals=None,
        receivers=np.array([], dtype=np.int32),
        senders=np.array([], dtype=np.int32),
        n_node=n_node.copy(),
        n_edge=np.array([0] * n_graph, dtype=np.int32),
    )
    graph = jraph.unpad_with_graphs(graph)
    graphs = jraph.unbatch_np(graph)
    new_graphs = []
    for g in graphs:
        pos = g.nodes.astype(np.float32)
        senders, receivers = matscipy.neighbours.neighbour_list(
            "ij", positions=pos, cutoff=float(cutoff), cell=np.eye(3)
        )
        g = jraph.GraphsTuple(
            nodes=None,
            edges=None,
            globals=None,
            receivers=receivers,
            senders=senders,
            n_node=g.n_node,
            n_edge=np.array([len(senders)], dtype=np.int32),
        )
        new_graphs.append(g)

    graph = jraph.batch_np(new_graphs)
    ok = np.array(np.sum(graph.n_edge) <= sum_n_edge, dtype=np.bool_)

    if not ok:
        dummy_graph = jraph.GraphsTuple(
            nodes=None,
            edges=None,
            globals=None,
            receivers=np.zeros(sum_n_edge, dtype=np.int32),
            senders=np.zeros(sum_n_edge, dtype=np.int32),
            n_node=np.array([sum_n_node] + [0] * (n_graph - 1), dtype=np.int32),
            n_edge=np.array([sum_n_edge] + [0] * (n_graph - 1), dtype=np.int32),
        )
        return (
            dummy_graph.senders,
            dummy_graph.receivers,
            dummy_graph.n_edge,
            ok,
        )

    graph = jraph.pad_with_graphs(
        graph, n_node=sum_n_node, n_edge=sum_n_edge, n_graph=n_graph
    )
    return (
        graph.senders,
        graph.receivers,
        graph.n_edge,
        ok,
    )


def nan_analysis(pred: datatypes.Predictions):
    global_attrs = [
        "stop_probs",
        "stop",
        "focus_indices",
        "target_species_logits",
        "target_species_probs",
        "target_species",
        "position_coeffs",
        "position_logits",
        "position_probs",
        "position_vectors",
    ]
    for attr in global_attrs:
        attr_value = getattr(pred.globals, attr)
        try:
            num_nans = jnp.isnan(attr_value).sum()
        except:
            try:
                num_nans = jnp.isnan(attr_value.array).sum()
            except:
                num_nans = jnp.isnan(attr_value.grid_values).sum()
        print(f"{attr} has {num_nans} nans")


def generate_molecules(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: int,
    seeds: Sequence[int],
    init_molecule: str,
    visualize: bool,
):
    """Generates molecules from a trained model at the given workdir."""
    # Create initial molecule, if provided.
    init_molecule, init_molecule_name = analysis.construct_molecule(init_molecule)
    logging.info(
        f"Initial molecule: {init_molecule.get_chemical_formula()} with numbers {init_molecule.numbers} and positions {init_molecule.positions}"
    )

    # Load model.
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )
    apply_fn = jax.jit(model.apply)
    logging.info(config.to_dict())

    # Create output directories.
    step_name = "step=best" if step == -1 else f"step={step}"
    molecules_outputdir = os.path.join(
        outputdir,
        name,
        f"fait={focus_and_atom_type_inverse_temperature}_pit={position_inverse_temperature}",
        "molecules",
        step_name,
    )
    os.makedirs(molecules_outputdir, exist_ok=True)
    if visualize:
        visualizations_outputdir = os.path.join(
            outputdir,
            name,
            f"fait={focus_and_atom_type_inverse_temperature}_pit={position_inverse_temperature}",
            "visualizations",
            "molecules",
            f"inverse_temperature={focus_and_atom_type_inverse_temperature},{position_inverse_temperature}",
            step_name,
        )
        os.makedirs(visualizations_outputdir, exist_ok=True)
    molecule_list = []

    def get_predictions(
        fragment: jraph.GraphsTuple, rng: chex.PRNGKey
    ) -> datatypes.Predictions:
        fragments = jraph.pad_with_graphs(fragment, n_node=80, n_edge=4096, n_graph=2)
        preds = apply_fn(
            params,
            rng,
            fragments,
            focus_and_atom_type_inverse_temperature,
            position_inverse_temperature,
        )

        # Remove the batch dimension.
        pred = jraph.unpad_with_graphs(preds)
        pred = pred._replace(
            globals=jax.tree_map(lambda x: np.squeeze(x, axis=0), pred.globals)
        )
        return pred

    def append_predictions(
        molecule: ase.Atoms, pred: datatypes.Predictions
    ) -> ase.Atoms:
        focus = pred.globals.focus_indices
        focus_position = molecule.positions[focus]
        new_position_relative_to_focus = pred.globals.position_vectors

        new_species = jnp.array(
            models.ATOMIC_NUMBERS[pred.globals.target_species.item()]
        )
        new_position = new_position_relative_to_focus + focus_position

        return ase.Atoms(
            positions=jnp.concatenate(
                [molecule.positions, new_position[None, :]], axis=0
            ),
            numbers=jnp.concatenate([molecule.numbers, new_species[None]], axis=0),
        )

    # Generate with different seeds.
    for seed in tqdm.tqdm(seeds, desc="Generating molecules"):
        rng = jax.random.PRNGKey(seed)
        molecule = init_molecule.copy()
        nan_found = False

        # Add atoms step-by-step.
        figs = []
        for step in range(MAX_NUM_ATOMS):
            step_rng, rng = jax.random.split(rng)
            fragment = input_pipeline.ase_atoms_to_jraph_graph(
                molecule, models.ATOMIC_NUMBERS, config.nn_cutoff
            )

            # Run the model on the current molecule.
            pred = get_predictions(fragment, step_rng)

            # Check for any NaNs in the predictions.
            num_nans = sum(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_map(lambda x: jnp.isnan(x).sum(), pred)
                )
            )
            if num_nans > 0:
                logging.info(
                    "NaNs in predictions at step %d. Stopping generation...", step
                )
                nan_found = True
                break

            # Save visualization of generation process.
            if visualize:
                fig = analysis.visualize_predictions(pred, molecule)
                figs.append(fig)

            # Check if we should stop.
            stop = pred.globals.stop.item()
            if stop:
                break

            # Append the new atom to the molecule.
            molecule = append_predictions(molecule, pred)

        # We don't generate molecules with more than MAX_NUM_ATOMS atoms.
        if molecule.numbers.shape[0] < 1000:
            logging.info("Generated %s", molecule.get_chemical_formula())
            if nan_found:
                outputfile = f"{init_molecule_name}_seed={seed}_NaN.xyz"
            else:
                outputfile = f"{init_molecule_name}_seed={seed}.xyz"
            ase.io.write(os.path.join(molecules_outputdir, outputfile), molecule)
            molecule_list.append(molecule)
        else:
            logging.info(
                "Discarding %s because it is too long", molecule.get_chemical_formula()
            )

        if visualize:
            # Combine visualizations.
            fig_all = analysis.combine_visualizations(figs)

            # Add title.
            model_name = analysis.get_title_for_name(name)
            fig_all.update_layout(
                title=f"{model_name}: Predictions for Seed {seed}",
                title_x=0.5,
            )

            # Save to file.
            outputfile = os.path.join(
                visualizations_outputdir,
                f"{init_molecule_name}_seed={seed}.html",
            )
            fig_all.write_html(outputfile, include_plotlyjs="cdn")

    # Save the generated molecules as an ASE database.
    output_db = os.path.join(
        molecules_outputdir, f"generated_molecules_init={init_molecule_name}.db"
    )
    with connect(output_db) as conn:
        for mol in molecule_list:
            conn.write(mol)


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    outputdir = FLAGS.outputdir
    focus_and_atom_type_inverse_temperature = (
        FLAGS.focus_and_atom_type_inverse_temperature
    )
    position_inverse_temperature = FLAGS.position_inverse_temperature
    step = FLAGS.step
    seeds = [int(seed) for seed in FLAGS.seeds]
    init = FLAGS.init
    visualize = FLAGS.visualize

    generate_molecules(
        workdir,
        outputdir,
        focus_and_atom_type_inverse_temperature,
        position_inverse_temperature,
        step,
        seeds,
        init,
        visualize,
    )


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for model.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses", "analysed_workdirs"),
        "Directory where molecules should be saved.",
    )
    flags.DEFINE_float(
        "focus_and_atom_type_inverse_temperature",
        1.0,
        "Inverse temperature value for sampling the focus and atom type.",
        short_name="fait",
    )
    flags.DEFINE_float(
        "position_inverse_temperature",
        1.0,
        "Inverse temperature value for sampling the position.",
        short_name="pit",
    )
    flags.DEFINE_integer(
        "step",
        -1,
        "Step number to load model from. The default of -1 corresponds to the best model.",
    )
    flags.DEFINE_list(
        "seeds",
        list(range(64)),
        "Seeds to attempt to generate molecules from.",
    )
    flags.DEFINE_string(
        "init",
        "C",
        "An initial molecular fragment to start the generation process from.",
    )
    flags.DEFINE_bool(
        "visualize",
        False,
        "Whether to visualize the generation process step-by-step.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
