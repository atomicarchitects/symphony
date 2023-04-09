"""Creates a series of visualizations to build up a molecule."""
import os
import pickle
import sys
from typing import Sequence, Tuple

import ase
import ase.build
import ase.data
import ase.io
import ase.visualize
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import numpy as np
import plotly.graph_objects as go
import plotly.subplots
import tqdm
import yaml
from absl import app, flags

sys.path.append("..")
import datatypes
import input_pipeline
import analyses.analysis as analysis
import models

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


def visualize_atom_removals(
    workdir: str, outputdir: str, beta: float, step: int, molecule_str: str
):
    """Generates visualizations of the predictions when removing each atom from a molecule."""
    molecule, molecule_name = analysis.construct_molecule(molecule_str)
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )

    apply_fn = jax.jit(model.apply)

    def get_predictions(
        frag: jraph.GraphsTuple,
    ) -> datatypes.Predictions:
        frags = jraph.pad_with_graphs(frag, n_node=32, n_edge=1024, n_graph=2)
        # We don't actually need a PRNG key, since we're not sampling.
        dummy_rng = jax.random.PRNGKey(0)
        preds = apply_fn(params, dummy_rng, frags, beta)
        pred = jraph.unpad_with_graphs(preds)
        pred = jax.tree_map(np.asarray, pred)
        pred = datatypes.Predictions(
            globals=jax.tree_map(lambda x: x.squeeze(axis=0), pred.globals),
            nodes=pred.nodes,
            edges=pred.edges,
            senders=pred.senders,
            receivers=pred.receivers,
            n_node=pred.n_node,
            n_edge=pred.n_edge,
        )
        return pred

    def remove_atom_and_visualize_predictions(
        molecule: ase.Atoms, target: int
    ) -> Tuple[go.Figure, go.Figure, go.Figure]:
        # Remove the target atom from the molecule.
        molecule_with_target_removed = ase.Atoms(
            positions=np.concatenate(
                [molecule.positions[:target], molecule.positions[target + 1 :]]
            ),
            numbers=np.concatenate(
                [molecule.numbers[:target], molecule.numbers[target + 1 :]]
            ),
        )
        frag = input_pipeline.ase_atoms_to_jraph_graph(
            molecule_with_target_removed,
            ATOMIC_NUMBERS,
            config.nn_cutoff,
        )
        pred = get_predictions(frag)

        # Make subplots.
        fig = plotly.subplots.make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            subplot_titles=("Molecule", "Target Element Probabilities"),
        )

        # Compute focus probabilities.
        focus = pred.globals.focus_indices.item()
        focus_position = frag.nodes.positions[focus]
        stop_prob = pred.globals.stop_probs.item()
        focus_probs = pred.nodes.focus_probs
        focus_probs = (1 - stop_prob) * focus_probs

        # Plot the actual molecule.
        mol_trace = []
        mol_trace.append(
            go.Scatter3d(
                x=molecule.positions[:, 0],
                y=molecule.positions[:, 1],
                z=molecule.positions[:, 2],
                mode="markers",
                marker=dict(
                    size=[ATOMIC_SIZES[num] for num in molecule.numbers],
                    color=[ATOMIC_COLORS[num] for num in molecule.numbers],
                ),
                hovertext=[
                    f"Element: {ase.data.chemical_symbols[num]}"
                    for num in molecule.numbers
                ],
                opacity=1.0,
                name="Molecule Atoms",
            )
        )

        # Highlight the focus probabilities.
        def get_scaling_factor(p: float) -> float:
            if p < 1 / len(molecule_with_target_removed):
                return 0.0
            return 1 + p**2

        def chosen_focus_string(i: int) -> str:
            if i == focus:
                return "(Chosen as Focus)"
            return "(Not Chosen as Focus)"

        mol_trace.append(
            go.Scatter3d(
                x=molecule_with_target_removed.positions[:, 0],
                y=molecule_with_target_removed.positions[:, 1],
                z=molecule_with_target_removed.positions[:, 2],
                mode="markers",
                marker=dict(
                    size=[
                        get_scaling_factor(focus_prob) * ATOMIC_SIZES[num]
                        for focus_prob, num in zip(
                            focus_probs, molecule_with_target_removed.numbers
                        )
                    ],
                    color=[
                        "rgba(150, 75, 0, 0.5)" for _ in molecule_with_target_removed
                    ],
                ),
                hovertext=[
                    f"Focus Probability: {focus_prob:.3f}<br>{chosen_focus_string(i)}"
                    for i, focus_prob in enumerate(focus_probs)
                ],
                name="Focus Probabilities",
            )
        )
        # Highlight the target atom.
        mol_trace.append(
            go.Scatter3d(
                x=[molecule.positions[target, 0]],
                y=[molecule.positions[target, 1]],
                z=[molecule.positions[target, 2]],
                mode="markers",
                marker=dict(
                    size=[1.05 * ATOMIC_SIZES[molecule.numbers[target]]],
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
            surface_r = go.Surface(
                **prob_r.plotly_surface(radius=RADII[i], translation=focus_position),
                colorscale=[
                    [0, "rgba(4, 59, 192, 0.)"],
                    [1, "rgba(4, 59, 192, 1.)"],
                ],
                showscale=False,
                cmin=cmin,
                cmax=cmax,
                name=f"Position Probabilities",
                legendgroup="Position Probabilities",
                showlegend=(i == 0),
            )
            mol_trace.append(surface_r)

        for trace in mol_trace:
            fig.add_trace(trace, row=1, col=1)

        # Plot target species probabilities.
        predicted_target_species = pred.globals.target_species.item()
        predicted_target_element = ELEMENTS[predicted_target_species]
        target_element_index = ATOMIC_NUMBERS.index(molecule.numbers[target])
        species_probs = pred.globals.target_species_probs.tolist()

        def chosen_element_string(elem: str) -> str:
            if elem == predicted_target_element:
                return "Chosen as Target Element"
            return "Not Chosen as Target Element"

        other_elements = (
            ELEMENTS[:target_element_index] + ELEMENTS[target_element_index + 1 :]
        )
        species_trace = [
            go.Bar(
                x=[ELEMENTS[target_element_index]],
                y=[species_probs[target_element_index]],
                hovertext=[chosen_element_string(ELEMENTS[target_element_index])],
                name="True Target Element Probability",
                marker=dict(color="green", opacity=0.8),
            ),
            go.Bar(
                x=other_elements,
                y=species_probs[:target_element_index]
                + species_probs[target_element_index + 1 :],
                hovertext=[chosen_element_string(elem) for elem in other_elements],
                name="Other Elements Probabilities",
                marker=dict(
                    color=[
                        ATOMIC_COLORS[ATOMIC_NUMBERS[ELEMENTS.index(elem)]]
                        for elem in other_elements
                    ],
                    opacity=0.8,
                ),
            ),
        ]

        fig.add_trace(species_trace[0], row=1, col=2)
        fig.add_trace(species_trace[1], row=1, col=2)

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

    # Create the output directory where HTML files will be saved.
    step_name = "step=best" if step == -1 else f"step={step}"
    outputdir = os.path.join(
        FLAGS.outputdir,
        "visualizations",
        "atom_removal",
        name,
        f"beta={beta}",
        step_name,
        step_name,
    )
    os.makedirs(outputdir, exist_ok=True)

    # Loop over all possible targets.
    for target in tqdm.tqdm(range(len(molecule)), desc="Targets"):
        fig = remove_atom_and_visualize_predictions(molecule, target=target)
        outputfile = os.path.join(
            outputdir,
            f"{molecule_name}_target={target}.html",
        )
        fig.write_html(outputfile)


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    outputdir = FLAGS.outputdir
    beta = FLAGS.beta
    step = FLAGS.step
    molecule_str = FLAGS.molecule

    visualize_atom_removals(workdir, outputdir, beta, step, molecule_str)


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

    flags.mark_flags_as_required(["workdir", "molecule"])
    app.run(main)
