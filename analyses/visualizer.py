from typing import Optional, Sequence
import plotly.graph_objects as go
import plotly.subplots
import numpy as np
import e3nn_jax as e3nn
import ase
import rdkit
import rdkit.Chem.PyMol
import os
import time

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


from symphony import models
from symphony import datatypes

ATOMIC_NUMBERS = models.ATOMIC_NUMBERS
ELEMENTS = ["O", "Si"]
NUMBER_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si"}
# ELEMENTS = ["H", "C", "N", "O", "F"]

# Colors and sizes for the atoms.
ATOMIC_COLORS = {
    1: "rgb(150, 150, 150)",  # H
    6: "rgb(50, 50, 50)",  # C
    7: "rgb(0, 100, 255)",  # N
    8: "rgb(255, 0, 0)",  # O
    9: "rgb(255, 0, 255)",  # F
    14: "rgb(180, 180, 180)",  # S
}
ATOMIC_SIZES = {
    1: 10,  # H
    6: 30,  # C
    7: 30,  # N
    8: 30,  # O
    9: 30,  # F
    14: 30,  # Si
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


def combine_visualizations(
    figs: Sequence[go.Figure],
    label_name: Optional[str] = None,
    labels: Optional[Sequence[str]] = None,
) -> go.Figure:
    """Combines multiple plotly figures generated by visualize_predictions() into one figure with a slider."""
    all_traces = []
    for fig in figs:
        all_traces.extend(fig.data)

    # Save the original visibility of the traces.
    original_visibility = {i: trace.visible for i, trace in enumerate(all_traces)}

    steps = []
    ct = 0
    start_indices = [0]
    for fig in figs:
        steps.append(
            dict(
                method="restyle",
                args=[
                    {
                        "visible": [
                            original_visibility[i]
                            if ct <= i < ct + len(fig.data)
                            else False
                            for i, trace in enumerate(all_traces)
                        ]
                    }
                ],
            )
        )
        ct += len(fig.data)
        start_indices.append(ct)

    if label_name is None:
        label_name = "steps"
    if labels is None:
        labels = steps

    axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title="",
        nticks=3,
    )
    layout = dict(
        sliders=[{label_name: labels}],
        title_x=0.5,
        width=1500,
        height=800,
        scene=dict(
            xaxis=dict(**axis),
            yaxis=dict(**axis),
            zaxis=dict(**axis),
            aspectmode="data",
        ),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.1,
        ),
    )

    fig_all = plotly.subplots.make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Input Fragment",
            "Predictions",
            "Focus and Atom Type Probabilities",
            "Stop Probability",
        ),
    )

    for i, trace in enumerate(all_traces):
        visible = original_visibility[i] if i < start_indices[1] else False
        trace.update(visible=visible)
        if trace.type in ["scatter3d", "surface"]:
            col = 1
        else:
            col = 2
        fig_all.add_trace(trace, row=1, col=col)
    fig_all.update_layout(layout)
    return fig_all


def get_plotly_traces_for_fragment(
    fragment: datatypes.Fragments,
) -> Sequence[go.Scatter3d]:
    """Returns the plotly traces for the fragment."""
    atomic_numbers = list(
        int(num) for num in models.get_atomic_numbers(fragment.nodes.species)
    )
    molecule_traces = []
    molecule_traces.append(
        go.Scatter3d(
            x=fragment.nodes.positions[:, 0],
            y=fragment.nodes.positions[:, 1],
            z=fragment.nodes.positions[:, 2],
            mode="markers",
            marker=dict(
                size=[ATOMIC_SIZES[num] for num in atomic_numbers],
                color=[ATOMIC_COLORS[num] for num in atomic_numbers],
            ),
            hovertext=[
                f"Element: {ase.data.chemical_symbols[num]}" for num in atomic_numbers
            ],
            opacity=1.0,
            name="Molecule Atoms",
            legendrank=1,
        )
    )
    # Add bonds.
    for i, j in zip(fragment.senders, fragment.receivers):
        molecule_traces.append(
            go.Scatter3d(
                x=fragment.nodes.positions[[i, j], 0],
                y=fragment.nodes.positions[[i, j], 1],
                z=fragment.nodes.positions[[i, j], 2],
                line=dict(color="black"),
                mode="lines",
                showlegend=False,
            )
        )

    # Highlight the target atom.
    if fragment.globals is not None:
        if fragment.globals.target_positions is not None and not fragment.globals.stop:
            # The target position is relative to the fragment's focus node.
            target_positions = (
                fragment.globals.target_positions + fragment.nodes.positions[0]
            )
            target_positions = target_positions.reshape(3)
            molecule_traces.append(
                go.Scatter3d(
                    x=[target_positions[0]],
                    y=[target_positions[1]],
                    z=[target_positions[2]],
                    mode="markers",
                    marker=dict(
                        size=[
                            1.05
                            * ATOMIC_SIZES[
                                models.ATOMIC_NUMBERS[
                                    fragment.globals.target_species.item()
                                ]
                            ]
                        ],
                        color=["green"],
                    ),
                    opacity=0.5,
                    name="Target Atom",
                )
            )

    return molecule_traces


def get_plotly_traces_for_predictions(
    pred: datatypes.Predictions, fragment: datatypes.Fragments
) -> Sequence[go.Scatter3d]:
    """Returns a list of plotly traces for the prediction."""

    atomic_numbers = list(
        int(num) for num in models.get_atomic_numbers(fragment.nodes.species)
    )
    focus = pred.globals.focus_indices.item()
    focus_position = fragment.nodes.positions[focus]
    focus_and_target_species_probs = pred.nodes.focus_and_target_species_probs
    focus_probs = focus_and_target_species_probs.sum(axis=-1)
    num_nodes, num_elements = focus_and_target_species_probs.shape

    # Highlight the focus probabilities, obtained by marginalization over all elements.
    def get_scaling_factor(focus_prob: float, num_nodes: int) -> float:
        """Returns a scaling factor for the size of the atom."""
        if focus_prob < 1 / num_nodes - 1e-3:
            return 0.95
        return 1 + focus_prob**2

    def chosen_focus_string(index: int, focus: int) -> str:
        """Returns a string indicating whether the atom was chosen as the focus."""
        if index == focus:
            return f"Atom {index} (Chosen as Focus)"
        return f"Atom {index} (Not Chosen as Focus)"

    molecule_traces = []
    molecule_traces.append(
        go.Scatter3d(
            x=fragment.nodes.positions[:, 0],
            y=fragment.nodes.positions[:, 1],
            z=fragment.nodes.positions[:, 2],
            mode="markers",
            marker=dict(
                size=[
                    get_scaling_factor(float(focus_prob), num_nodes) * ATOMIC_SIZES[num]
                    for focus_prob, num in zip(focus_probs, atomic_numbers)
                ],
                color=["rgba(150, 75, 0, 0.5)" for _ in range(num_nodes)],
            ),
            hovertext=[
                f"Focus Probability: {focus_prob:.3f}<br>{chosen_focus_string(i, focus)}"
                for i, focus_prob in enumerate(focus_probs)
            ],
            name="Focus Probabilities",
        )
    )

    # Highlight predicted position, if not stopped.
    if not pred.globals.stop:
        predicted_target_position = focus_position + pred.globals.position_vectors
        molecule_traces.append(
            go.Scatter3d(
                x=[predicted_target_position[0]],
                y=[predicted_target_position[1]],
                z=[predicted_target_position[2]],
                mode="markers",
                marker=dict(
                    size=[
                        1.05
                        * ATOMIC_SIZES[
                            models.ATOMIC_NUMBERS[pred.globals.target_species.item()]
                        ]
                    ],
                    color=["purple"],
                ),
                opacity=0.5,
                name="Predicted Atom",
            )
        )

    # Since we downsample the position grid, we need to recompute the position probabilities.
    position_coeffs = pred.globals.log_position_coeffs
    radii = pred.globals.radial_bins
    num_radii = radii.shape[0]
    position_logits = models.log_coeffs_to_logits(position_coeffs, 50, 99, num_radii)
    position_logits.grid_values -= np.max(position_logits.grid_values)
    position_probs = position_logits.apply(np.exp)

    count = 0
    cmin = 0.0
    cmax = position_probs.grid_values.max().item()
    for i in range(len(radii)):
        prob_r = position_probs[i]

        # Skip if the probability is too small.
        if prob_r.grid_values.max() < 1e-2 * cmax:
            continue

        count += 1
        surface_r = go.Surface(
            **prob_r.plotly_surface(radius=radii[i], translation=focus_position),
            colorscale=[[0, "rgba(4, 59, 192, 0.)"], [1, "rgba(4, 59, 192, 1.)"]],
            showscale=False,
            cmin=cmin,
            cmax=cmax,
            name="Position Probabilities",
            legendgroup="Position Probabilities",
            showlegend=(count == 1),
            visible="legendonly",
        )
        molecule_traces.append(surface_r)

    # Plot spherical harmonic projections of logits.
    # Find closest index in RADII to the sampled positions.
    radii = pred.globals.radial_bins
    radius = np.linalg.norm(pred.globals.position_vectors, axis=-1)
    most_likely_radius_index = np.abs(radii - radius).argmin()
    most_likely_radius = radii[most_likely_radius_index]
    all_sigs = e3nn.to_s2grid(
        position_coeffs, 50, 99, quadrature="soft", p_val=1, p_arg=-1
    )
    cmin = all_sigs.grid_values.min().item()
    cmax = all_sigs.grid_values.max().item()
    for channel in range(position_coeffs.shape[0]):
        most_likely_radius_coeffs = position_coeffs[channel, most_likely_radius_index]
        most_likely_radius_sig = e3nn.to_s2grid(
            most_likely_radius_coeffs, 50, 99, quadrature="soft", p_val=1, p_arg=-1
        )
        spherical_harmonics = go.Surface(
            most_likely_radius_sig.plotly_surface(
                scale_radius_by_amplitude=True,
                radius=most_likely_radius,
                translation=focus_position,
                normalize_radius_by_max_amplitude=True,
            ),
            cmin=cmin,
            cmax=cmax,
            name=f"Spherical Harmonics for Logits: Channel {channel}",
            showlegend=True,
            visible="legendonly",
        )
        molecule_traces.append(spherical_harmonics)

    # Plot target species probabilities.
    stop_probability = pred.globals.stop_probs.item()
    predicted_target_species = pred.globals.target_species.item()
    if fragment.globals is not None and not fragment.globals.stop:
        true_focus = 0  # This is a convention used in our training pipeline.
        true_target_species = fragment.globals.target_species.item()
    else:
        true_focus = None
        true_target_species = None

    # We highlight the true target if provided.
    def get_focus_string(atom_index: int) -> str:
        """Get the string for the focus."""
        base_string = f"Atom {atom_index}"
        if atom_index == focus:
            base_string = f"{base_string}<br>Predicted Focus"
        if atom_index == true_focus:
            base_string = f"{base_string}<br>True Focus"
        return base_string

    def get_atom_type_string(element_index: int, element: str) -> str:
        """Get the string for the atom type."""
        base_string = f"{element}"
        if element_index == predicted_target_species:
            base_string = f"{base_string}<br>Predicted Species"
        if element_index == true_target_species:
            base_string = f"{base_string}<br>True Species"
        return base_string

    focus_and_atom_type_traces = [
        go.Heatmap(
            x=[
                get_atom_type_string(index, elem)
                for index, elem in enumerate(ELEMENTS[:num_elements])
            ],
            y=[get_focus_string(i) for i in range(num_nodes)],
            z=np.round(pred.nodes.focus_and_target_species_probs, 3),
            texttemplate="%{z}",
            showlegend=False,
            showscale=False,
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            xgap=1,
            ygap=1,
        ),
    ]
    stop_traces = [
        go.Bar(
            x=["STOP"],
            y=[stop_probability],
            showlegend=False,
        )
    ]
    return molecule_traces, focus_and_atom_type_traces, stop_traces


def visualize_fragment(
    fragment: datatypes.Fragments,
) -> go.Figure:
    """Visualizes the predictions for a molecule at a particular step."""
    # Make subplots.
    fig = plotly.subplots.make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scene"}]],
        subplot_titles=("Input Fragment",),
    )

    # Traces corresponding to the input fragment.
    fragment_traces = get_plotly_traces_for_fragment(fragment)
    for trace in fragment_traces:
        fig.add_trace(trace, row=1, col=1)

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
        scene=dict(
            xaxis=dict(**axis),
            yaxis=dict(**axis, scaleanchor="x", scaleratio=1),
            zaxis=dict(**axis, scaleanchor="x", scaleratio=1),
            aspectmode="data",
        ),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.1,
        ),
    )

    try:
        return go.FigureWidget(fig)
    except NotImplementedError:
        return fig


def visualize_predictions(
    pred: datatypes.Predictions,
    fragment: datatypes.Fragments,
) -> go.Figure:
    """Visualizes the predictions for a molecule at a particular step."""

    # Make subplots.
    fig = plotly.subplots.make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "xy"}, {"type": "xy"}]],
        column_widths=[0.3, 0.3, 0.2, 0.1],
        subplot_titles=("Input Fragment", "Predictions", "", ""),
    )

    # Traces corresponding to the input fragment.
    fragment_traces = get_plotly_traces_for_fragment(fragment)

    # Traces corresponding to the prediction.
    (
        predicted_fragment_traces,
        focus_and_atom_type_traces,
        stop_traces,
    ) = get_plotly_traces_for_predictions(pred, fragment)

    for trace in fragment_traces:
        fig.add_trace(trace, row=1, col=1)
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    for trace in predicted_fragment_traces:
        fig.add_trace(trace, row=1, col=2)

    for trace in focus_and_atom_type_traces:
        fig.add_trace(trace, row=1, col=3)

    for trace in stop_traces:
        fig.add_trace(trace, row=1, col=4)

    # Update the layout.
    centre_of_mass = np.mean(fragment.nodes.positions, axis=0)
    furthest_dist = np.max(
        np.linalg.norm(
            fragment.nodes.positions + pred.globals.position_vectors - centre_of_mass,
            axis=-1,
        )
    )
    furthest_dist = np.max(
        np.linalg.norm(
            fragment.nodes.positions - pred.globals.position_vectors - centre_of_mass,
            axis=-1,
        )
    )
    min_range = centre_of_mass - furthest_dist
    max_range = centre_of_mass + furthest_dist
    axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        title="",
        nticks=3,
    )
    fig.update_layout(
        scene1=dict(
            xaxis=dict(**axis, range=[min_range[0], max_range[0]]),
            yaxis=dict(**axis, range=[min_range[1], max_range[1]]),
            zaxis=dict(**axis, range=[min_range[2], max_range[2]]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        scene2=dict(
            xaxis=dict(**axis, range=[min_range[0], max_range[0]]),
            yaxis=dict(**axis, range=[min_range[1], max_range[1]]),
            zaxis=dict(**axis, range=[min_range[2], max_range[2]]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
        ),
        yaxis2=dict(
            range=[0, 1],
        ),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.1,
        ),
    )

    # Sync cameras.
    try:
        fig_widget = go.FigureWidget(fig)

        def cam_change_1(layout, camera):
            fig_widget.layout.scene2.camera = camera

        def cam_change_2(layout, camera):
            if fig_widget.layout.scene1.camera != camera:
                fig_widget.layout.scene1.camera = camera

        fig_widget.layout.scene1.on_change(cam_change_1, "camera")
        fig_widget.layout.scene2.on_change(cam_change_2, "camera")

        return fig_widget
    except NotImplementedError:
        return fig


def visualize_bonded_molecules(input_sdf_dir: str, output_dir: str) -> None:
    """Visualizes the molecules in the input SDF directory using RDKit and PyMol."""

    # 2D visualization of molecules, as a grid.
    os.makedirs(os.path.join(output_dir, "2D"), exist_ok=True)
    molecules = []
    filenames = []
    for molecules_file in os.listdir(input_sdf_dir):
        if not molecules_file.endswith(".sdf"):
            continue

        molecule = rdkit.Chem.MolFromMolFile(
            os.path.join(input_sdf_dir, molecules_file)
        )
        filenames.append(molecules_file)
        molecules.append(molecule)

    rdkit.Chem.Draw.MolsToGridImage(
        molecules,
        molsPerRow=6,
        subImgSize=(200, 200),
        legends=[
            molecules_file.replace(".sdf", "")
            for molecules_file in os.listdir(input_sdf_dir)
        ],
    ).save(os.path.join(output_dir, "2D", "molecules.png"))

    # 3D visualization of molecules.
    os.makedirs(os.path.join(output_dir, "3D"), exist_ok=True)
    viewer = rdkit.Chem.PyMol.MolViewer()
    for molecules_file in os.listdir(input_sdf_dir):
        if not molecules_file.endswith(".sdf"):
            continue

        viewer.DeleteAll()
        viewer.server.do("reset")
        viewer.LoadFile(
            os.path.join(input_sdf_dir, molecules_file), name=molecules_file
        )
        viewer.Zoom(molecules_file)
        viewer.server.do("orient")
        viewer.server.do("bg_color white")
        viewer.server.do("util.cbaw")
        viewer.server.do("color gray, (name C*)")
        viewer.server.do("center")
        viewer.server.do("zoom center, 4")
        viewer.server.do("hide lines")
        viewer.server.do("show sticks")
        viewer.server.do("show spheres")
        viewer.server.do("set stick_radius, 0.1, (all)")
        viewer.server.do("set sphere_scale, 0.25, (all)")

        image = viewer.GetPNG(3000, 2000)
        image.save(
            os.path.join(output_dir, "3D", molecules_file.replace(".sdf", ".png"))
        )
        image.close()

        # Some delay for rendering.
        time.sleep(1)
