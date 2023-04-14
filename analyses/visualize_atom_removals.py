"""Creates a series of visualizations to build up a molecule."""
import os
import pickle
import sys
from typing import Optional, Sequence

import ase
import ase.build
import ase.data
import ase.io
import ase.visualize
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import plotly.graph_objects as go
import plotly.subplots
import tqdm
from absl import app, flags, logging

sys.path.append("..")

import analyses.analysis as analysis  # noqa: E402
import datatypes  # noqa: E402
import input_pipeline  # noqa: E402
import models  # noqa: E402

FLAGS = flags.FLAGS
ATOMIC_NUMBERS = models.ATOMIC_NUMBERS
ELEMENTS = ["H", "C", "N", "O", "F"]
RADII = models.RADII

# Colors and sizes for the atoms.
ATOMIC_COLORS = {
    1: "rgb(150, 150, 150)",  # H
    6: "rgb(50, 50, 50)",  # C
    7: "rgb(0, 100, 255)",  # N
    8: "rgb(255, 0, 0)",  # O
    9: "rgb(255, 0, 255)",  # F
}
ATOMIC_SIZES = {
    1: 10,  # H
    6: 30,  # C
    7: 30,  # N
    8: 30,  # O
    9: 30,  # F
}


def get_title_for_name(name: str) -> str:
    """Returns the title for the given name."""
    if "e3schnet" in name:
        return "E3SchNet"
    elif "mace" in name:
        return "MACE"
    elif "nequip" in name:
        return "NequIP"
    return name.title()


def visualize_predictions(
    pred: datatypes.Predictions,
    input_molecule: ase.Atoms,
    molecule_with_target: Optional[ase.Atoms] = None,
    target: Optional[int] = None,
) -> go.Figure:
    """Visualizes the predictions for a molecule with a target atom removed."""

    def get_scaling_factor(focus_prob: float, num_nodes: int) -> float:
        """Returns a scaling factor for the size of the atom."""
        if focus_prob < 1 / num_nodes - 1e-3:
            return 0.95
        return 1 + focus_prob**2

    def chosen_focus_string(index: int, focus: int) -> str:
        """Returns a string indicating whether the atom was chosen as the focus."""
        if index == focus:
            return "(Chosen as Focus)"
        return "(Not Chosen as Focus)"

    def chosen_element_string(element: str, predicted_target_element: str) -> str:
        """Returns a string indicating whether an element was chosen as the target element."""
        if element == predicted_target_element:
            return "Chosen as Target Element"
        return "Not Chosen as Target Element"

    # Make subplots.
    fig = plotly.subplots.make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("Molecule", "Target Element Probabilities"),
    )

    # Highlight the focus probabilities.
    mol_trace = []
    focus = pred.globals.focus_indices.item()
    focus_position = input_molecule.positions[focus]
    stop_prob = pred.globals.stop_probs.item()
    focus_probs = (1 - stop_prob) * pred.nodes.focus_probs

    mol_trace.append(
        go.Scatter3d(
            x=input_molecule.positions[:, 0],
            y=input_molecule.positions[:, 1],
            z=input_molecule.positions[:, 2],
            mode="markers",
            marker=dict(
                size=[
                    get_scaling_factor(float(focus_prob), len(input_molecule))
                    * ATOMIC_SIZES[num]
                    for focus_prob, num in zip(focus_probs, input_molecule.numbers)
                ],
                color=["rgba(150, 75, 0, 0.5)" for _ in input_molecule],
            ),
            hovertext=[
                f"Focus Probability: {focus_prob:.3f}<br>{chosen_focus_string(i, focus)}"
                for i, focus_prob in enumerate(focus_probs)
            ],
            name="Focus Probabilities",
        )
    )
    # Plot the actual molecule.
    mol_trace.append(
        go.Scatter3d(
            x=input_molecule.positions[:, 0],
            y=input_molecule.positions[:, 1],
            z=input_molecule.positions[:, 2],
            mode="markers",
            marker=dict(
                size=[ATOMIC_SIZES[num] for num in input_molecule.numbers],
                color=[ATOMIC_COLORS[num] for num in input_molecule.numbers],
            ),
            hovertext=[
                f"Element: {ase.data.chemical_symbols[num]}"
                for num in input_molecule.numbers
            ],
            opacity=1.0,
            name="Molecule Atoms",
            legendrank=1,
        )
    )
    # Highlight the target atom.
    if target is not None:
        mol_trace.append(
            go.Scatter3d(
                x=[molecule_with_target.positions[target, 0]],
                y=[molecule_with_target.positions[target, 1]],
                z=[molecule_with_target.positions[target, 2]],
                mode="markers",
                marker=dict(
                    size=[1.05 * ATOMIC_SIZES[molecule_with_target.numbers[target]]],
                    color=["green"],
                ),
                opacity=0.5,
                name="Target Atom",
            )
        )

    # Since we downsample the position grid, we need to recompute the focus probabilities.
    position_logits = e3nn.to_s2grid(
        pred.globals.position_coeffs,
        50,
        99,
        quadrature="gausslegendre",
        normalization="integral",
        p_val=1,
        p_arg=-1,
    )
    position_probs = position_logits.apply(
        lambda x: jnp.exp(x - position_logits.grid_values.max())
    )
    cmin = 0.0
    cmax = position_probs.grid_values.max().item()
    for i in range(len(RADII)):
        prob_r = position_probs[i]

        # Skip if the probability is too small.
        if prob_r.grid_values.max() < 1e-2 * cmax:
            continue

        surface_r = go.Surface(
            **prob_r.plotly_surface(radius=RADII[i], translation=focus_position),
            colorscale=[[0, "rgba(4, 59, 192, 0.)"], [1, "rgba(4, 59, 192, 1.)"]],
            showscale=False,
            cmin=cmin,
            cmax=cmax,
            name="Position Probabilities",
            legendgroup="Position Probabilities",
            showlegend=(i == 0),
        )
        mol_trace.append(surface_r)

    for trace in mol_trace:
        fig.add_trace(trace, row=1, col=1)

    # Plot target species probabilities.
    predicted_target_species = pred.globals.target_species.item()
    predicted_target_element = ELEMENTS[predicted_target_species]
    species_probs = pred.globals.target_species_probs.tolist()

    # We highlight the true target if provided.
    if target is not None:
        target_element_index = ATOMIC_NUMBERS.index(
            molecule_with_target.numbers[target]
        )
        other_elements = (
            ELEMENTS[:target_element_index] + ELEMENTS[target_element_index + 1 :]
        )
        species_trace = [
            go.Bar(
                x=[ELEMENTS[target_element_index]],
                y=[species_probs[target_element_index]],
                hovertext=[
                    chosen_element_string(
                        ELEMENTS[target_element_index], predicted_target_element
                    )
                ],
                name="True Target Element Probability",
                marker=dict(color="green", opacity=0.8),
                showlegend=False,
            ),
            go.Bar(
                x=other_elements,
                y=species_probs[:target_element_index]
                + species_probs[target_element_index + 1 :],
                hovertext=[
                    chosen_element_string(elem, predicted_target_element)
                    for elem in other_elements
                ],
                name="Other Elements Probabilities",
                marker=dict(
                    color=[
                        "gray" if elem != predicted_target_element else "blue"
                        for elem in other_elements
                    ],
                    opacity=0.8,
                ),
                showlegend=False,
            ),
        ]

    else:
        species_trace = [
            go.Bar(
                x=ELEMENTS,
                y=species_probs,
                name="Elements Probabilities",
                hovertext=[
                    chosen_element_string(elem, predicted_target_element)
                    for elem in ELEMENTS
                ],
                marker=dict(
                    color=[
                        "gray" if elem != predicted_target_element else "blue"
                        for elem in ELEMENTS
                    ],
                    opacity=0.8,
                ),
                showlegend=False,
            ),
        ]

    for trace in species_trace:
        fig.add_trace(trace, row=1, col=2)

    # Update the layout.
    axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title="",
        nticks=3,
    )
    fig.update_layout(
        width=1500,
        height=800,
        scene=dict(
            xaxis=dict(**axis),
            yaxis=dict(**axis),
            zaxis=dict(**axis),
            aspectmode="data",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.1,
        ),
    )
    return fig


def visualize_atom_removals(
    workdir: str,
    outputdir: str,
    beta: float,
    step: int,
    molecule_str: str,
    use_cache: bool,
    seed: int,
):
    """Generates visualizations of the predictions when removing each atom from a molecule."""
    molecule, molecule_name = analysis.construct_molecule(molecule_str)
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )

    # Remove the target atoms from the molecule.
    molecules_with_target_removed = []
    fragments = []
    for target in range(len(molecule)):
        molecule_with_target_removed = ase.Atoms(
            positions=np.concatenate(
                [molecule.positions[:target], molecule.positions[target + 1 :]]
            ),
            numbers=np.concatenate(
                [molecule.numbers[:target], molecule.numbers[target + 1 :]]
            ),
        )
        fragment = input_pipeline.ase_atoms_to_jraph_graph(
            molecule_with_target_removed,
            ATOMIC_NUMBERS,
            config.nn_cutoff,
        )

        molecules_with_target_removed.append(molecule_with_target_removed)
        fragments.append(fragment)

    # We don't actually need a PRNG key, since we're not sampling.
    logging.info("Computing predictions...")
    preds_path = os.path.join(
        f"cached/{workdir.replace('/', '_')}_{molecule_name}_preds.pkl"
    )
    if use_cache and os.path.exists(preds_path):
        logging.info("Using cached predictions at %s", os.path.abspath(preds_path))
        preds = pickle.load(open(preds_path, "rb"))
    else:
        rng = jax.random.PRNGKey(seed)
        preds = jax.jit(model.apply)(params, rng, jraph.batch(fragments), beta)
        preds = jax.tree_map(np.asarray, preds)
        preds = jraph.unbatch(preds)
        os.makedirs(os.path.dirname(preds_path), exist_ok=True)
        pickle.dump(preds, open(preds_path, "wb"))
        logging.info("Predictions computed.")

    # Create the output directory where HTML files will be saved.
    step_name = "step=best" if step == -1 else f"step={step}"
    outputdir = os.path.join(
        FLAGS.outputdir,
        "visualizations",
        "atom_removal",
        name,
        f"beta={beta}",
        step_name,
    )
    os.makedirs(outputdir, exist_ok=True)

    # Loop over all possible targets.
    logging.info("Visualizing predictions...")
    for target in tqdm.tqdm(range(len(molecule)), desc="Targets"):
        # We have to remove the batch dimension.
        # Also, correct the focus indices due to batching.
        pred = preds[target]._replace(
            globals=jax.tree_map(lambda x: np.squeeze(x, axis=0), preds[target].globals)
        )
        corrected_focus_indices = pred.globals.focus_indices - sum(
            p.n_node.item() for i, p in enumerate(preds) if i < target
        )
        pred = pred._replace(
            globals=pred.globals._replace(focus_indices=corrected_focus_indices)
        )

        # Visualize and add a title.
        fig = visualize_predictions(
            pred, molecules_with_target_removed[target], molecule, target
        )
        model_name = get_title_for_name(name)
        fig.update_layout(
            title=f"{model_name}: Predictions for {molecule_name}",
            title_x=0.5,
        )

        # Save to file.
        outputfile = os.path.join(
            outputdir,
            f"{molecule_name}_seed={seed}_target={target}.html",
        )
        fig.write_html(outputfile, include_plotlyjs="cdn")


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    outputdir = FLAGS.outputdir
    beta = FLAGS.beta
    step = FLAGS.step
    molecule_str = FLAGS.molecule
    use_cache = FLAGS.use_cache
    seed = FLAGS.seed

    visualize_atom_removals(
        workdir, outputdir, beta, step, molecule_str, use_cache, seed
    )


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for model.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses"),
        "Directory where visualizations should be saved.",
    )
    flags.DEFINE_float("beta", 1.0, "Inverse temperature value for sampling.")
    flags.DEFINE_integer(
        "step",
        -1,
        "Step number to load model from. The default of -1 corresponds to the best model.",
    )
    flags.DEFINE_string(
        "molecule",
        None,
        "Molecule to use for experiment. Can be specified either as an index for the QM9 dataset, a name for ase.build.molecule(), or a file with atomic numbers and coordinates for ase.io.read().",
    )
    flags.DEFINE_bool(
        "use_cache",
        False,
        "Whether to use cached predictions if they exist.",
    )
    flags.DEFINE_integer(
        "seed",
        0,
        "PRNG seed for sampling.",
    )

    flags.mark_flags_as_required(["workdir", "molecule"])
    app.run(main)
