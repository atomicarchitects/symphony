"""Creates a series of visualizations to build up a molecule."""
import os
import pickle
import sys

from absl import flags
from absl import app
import ase
import ase.build
import ase.data
import ase.io
import ase.visualize
import jax
import jax.numpy as jnp
import jraph
import ml_collections
import tqdm
import chex
import yaml
import plotly.graph_objects as go
import e3nn_jax as e3nn
import numpy as np


sys.path.append("..")
import datatypes
import input_pipeline
import analyses.analysis as analysis
import models
import qm9

FLAGS = flags.FLAGS
ATOMIC_NUMBERS = models.ATOMIC_NUMBERS
RADII = models.RADII


def get_molecule(molecule_str: str) -> ase.Atoms:
    # A number is interpreted as a QM9 molecule index.
    if molecule_str.isdigit():
        dataset = qm9.load_qm9("qm9_data")
        return dataset[int(molecule_str)]

    # If the string is a valid molecule name, try to build it.
    try:
        return ase.build.molecule(molecule_str)
    except ValueError:
        return ase.io.read(molecule_str)


def main():
    workdir = os.path.abspath(FLAGS.workdir)
    outputdir = FLAGS.outputdir
    beta = FLAGS.beta
    step_num = FLAGS.step_num
    molecule_str = FLAGS.molecule

    if step_num == -1:
        params_file = os.path.join(workdir, "checkpoints/params_best.pkl")
        step_name = "step=best"
    else:
        params_file = os.path.join(workdir, "checkpoints/params_{step_num}.pkl")
        step_name = f"step={step_num}"

    try:
        with open(params_file, "rb") as f:
            params = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find params file {params_file}")

    with open(workdir + "/config.yml", "rt") as config_file:
        config = yaml.unsafe_load(config_file)
    assert config is not None
    config = ml_collections.ConfigDict(config)

    name = analysis.name_from_workdir(workdir)
    model = models.create_model(config, run_in_evaluation_mode=True)
    apply_fn = jax.jit(model.apply)

    molecule = get_molecule(molecule_str)

    def get_predictions(
        frag: jraph.GraphsTuple, rng: chex.PRNGKey
    ) -> datatypes.Predictions:
        frags = jraph.pad_with_graphs(frag, n_node=32, n_edge=1024, n_graph=2)
        preds = apply_fn(params, rng, frags, beta)
        pred = jraph.unpad_with_graphs(preds)
        return pred

    def remove_atom_and_visualize_predictions(
        molecule: ase.Atoms, target: int
    ) -> go.Figure:
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

        # We don't actually need a PRNG key, since we're not sampling.
        dummy_rng = jax.random.PRNGKey(0)
        pred = get_predictions(frag, dummy_rng, beta)

        ATOMIC_COLORS = {
            1: "rgb(200, 200, 200)",  # H
            6: "rgb(50, 50, 50)",  # C
            7: "rgb(0, 100, 255)",  # N
            8: "rgb(255, 0, 0)",  # O
            9: "rgb(255, 0, 255)",  # F
        }
        ATOMIC_SIZE = {
            1: 10,  # H
            6: 30,  # C
            7: 30,  # N
            8: 30,  # O
            9: 30,  # F
        }

        figdata = []
        figdata.append(
            go.Scatter3d(
                x=molecule.positions[:, 0],
                y=molecule.positions[:, 1],
                z=molecule.positions[:, 2],
                mode="markers",
                marker=dict(
                    size=[ATOMIC_SIZE[i] for i in z],
                    color=[ATOMIC_COLORS[i] for i in z],
                ),
                hovertext=[ase.data.chemical_symbols[i] for i in z],
                opacity=1.0,
                showlegend=False,
            )
        )
        figdata.append(
            go.Scatter3d(
                x=[molecule.positions[target, 0]],
                y=[molecule.positions[target, 1]],
                z=[molecule.positions[target, 2]],
                mode="markers",
                marker=dict(
                    size=1.05 * ATOMIC_SIZE[molecule.numbers[target]],
                    color="yellow",
                ),
                opacity=0.5,
                name="Target",
            )
        )

        focus = pred.globals.focus_indices[0]
        sp = pred.globals.target_species.item()
        position_probs = pred.globals.position_probs
        position_probs = position_probs.resample(50, 99, 6)
        pos = frag.nodes.positions[focus]

        cmax = position_probs.grid_values.max().item()
        for i in range(len(RADII)):
            prob_r = position_probs.grid_values[0, i]
            prob_r = e3nn.SphericalSignal(prob_r, position_probs.quadrature)

            surface_r = go.Surface(
                **prob_r.plotly_surface(radius=RADII[i], translation=pos),
                colorscale=[
                    [0, f"rgba(0, 0, 0, 0.0)"],
                    [1, f"rgba(0, 0, 0, 1.0)"],
                ],
                showscale=False,
                cmin=0.0,
                cmax=cmax,
                name=f"Prediction: {ase.data.chemical_symbols[ATOMIC_NUMBERS[sp]]}",
            )
            figdata.append(surface_r)

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

        return go.Figure(data=figdata, layout=layout)

    # Loop over all possible targets.
    molecule_name = molecule.get_chemical_formula()
    for target in tqdm.tqdm(len(molecule)):
        fig = remove_atom_and_visualize_predictions(molecule, target=target)
        outputfile = os.path.join(
            outputdir,
            "visualizations",
            "atom_removal",
            name,
            f"beta={beta}",
            step_name,
            f"{molecule_name()}_target={target}.html",
        )
        fig.write_html(outputfile)


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
