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
import jraph
import ml_collections
import numpy as np
import plotly.graph_objects as go
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

        # Compute focus probabilities.
        num_nodes = len(molecule)
        stop_probs = pred.globals.stop_probs.item()
        focus = pred.globals.focus_indices.item()
        focus_position = frag.nodes.positions[focus]
        focus_probs_shifted = np.concatenate([pred.nodes.focus_probs[:target], [0], pred.nodes.focus_probs[target:]])
        focus_probs_renormalized = focus_probs_shifted * (1 - stop_probs)

        # Plot species probabilities.
        target_element_index = ATOMIC_NUMBERS.index(molecule.numbers[target])
        species_probs = pred.globals.target_species_probs[0].tolist()
        species_fig = go.Figure(
            [
                go.Bar(
                    x=[ELEMENTS[target_element_index]],
                    y=[species_probs[target_element_index]],
                ),
                go.Bar(
                    x=ELEMENTS[:target_element_index]
                    + ELEMENTS[target_element_index + 1 :],
                    y=species_probs[:target_element_index]
                    + species_probs[target_element_index + 1 :],
                ),
            ]
        )

        # Plot the actual molecule.
        mol_fig_data = []
        mol_fig_data.append(
            go.Scatter3d(
                x=molecule.positions[:, 0],
                y=molecule.positions[:, 1],
                z=molecule.positions[:, 2],
                mode="markers",
                marker=dict(
                    size=[ATOMIC_SIZES[i] for i in molecule.numbers],
                    color=[ATOMIC_COLORS[i] for i in molecule.numbers],
                ),
                hovertext=[ase.data.chemical_symbols[i] for i in molecule.numbers],
                text=["p(focus) = {:.2f}".format(focus_probs_renormalized[i]) if i != target else "Removed" for i in range(num_nodes)],
                opacity=1.0,
                name="Molecule",
            )
        )
        # Highlight the target atom.
        mol_fig_data.append(
            go.Scatter3d(
                x=[molecule.positions[target, 0], focus_pos[0]],
                y=[molecule.positions[target, 1], focus_pos[1]],
                z=[molecule.positions[target, 2], focus_pos[2]],
                mode="markers",
                marker=dict(
                    size=[
                        1.3 * ATOMIC_SIZE[molecule.numbers[target]],
                        1.3 * ATOMIC_SIZE[ATOMIC_NUMBERS[focus_sp]],
                    ],
                    sizemode="diameter",
                    color=["yellow", "green"],
                    size=1.05 * ATOMIC_SIZES[molecule.numbers[target]],
                    color="green",
                ),
                opacity=0.5,
                name="Target",
            )   
        )

        target_sp = pred.globals.target_species.item()
        target_color = ATOMIC_COLORS[ATOMIC_NUMBERS[target_sp]]
        target_logits = e3nn.to_s2grid(
            pred.globals.position_coeffs,
            50,
            99,
            quadrature="gausslegendre",
            normalization="integral",
            p_val=1,
            p_arg=-1,
        )
        target_probs = target_logits.apply(
            lambda x: jnp.exp(x - target_logits.grid_values.max())
        )

        cmin = 0.0
        cmax = target_probs.grid_values.max().item()
        for i in range(len(RADII)):
            p = target_probs[0, i]

            if p.grid_values.max() < cmax / 100.0:
                continue

            figdata.append(
                go.Surface(
                    **p.plotly_surface(radius=RADII[i], translation=focus_pos),
                    colorscale=[
                        [0, f"rgba({target_color[4:-1]}, 0.0)"],
                        [1, f"rgba({target_color[4:-1]}, 1.0)"],
                    ],
                    showscale=False,
                    cmin=cmin,
                    cmax=cmax,
                    name=f"Prediction: {ase.data.chemical_symbols[ATOMIC_NUMBERS[target_sp]]}",
                )
            )
            mol_fig_data.append(surface_r)

        axis = dict(
            showbackground=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title="",
            nticks=3,
        )

        layout = go.Layout(
            width=1200,
            height=800,
            scene=dict(
                xaxis=dict(**axis),
                yaxis=dict(**axis),
                zaxis=dict(**axis),
                aspectmode="data",
                camera=dict(
                    up=dict(x=0, y=1, z=0),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=0, z=5),
                    projection=dict(type="orthographic"),
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        mol_fig = go.Figure(data=mol_fig_data, layout=layout)

        return focus_fig, species_fig, mol_fig

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
        focus_fig, species_fig, mol_fig = remove_atom_and_visualize_predictions(
            molecule, target=target
        )
        outputfile = os.path.join(
            outputdir,
            f"{molecule_name}_target={target}_focus.png",
        )
        focus_fig.write_image(outputfile)

        outputfile = os.path.join(
            outputdir,
            f"{molecule_name}_target={target}_species.png",
        )
        species_fig.write_image(outputfile)

        outputfile = os.path.join(
            outputdir,
            f"{molecule_name}_target={target}_molecule.png",
        )
        mol_fig.write_image(outputfile)

        outputfile = os.path.join(
            outputdir,
            f"{molecule_name}_target={target}_molecule.html",
        )
        mol_fig.write_html(outputfile)




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
