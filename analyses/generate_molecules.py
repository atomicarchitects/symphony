"""Generates molecules from a trained model."""

from typing import Sequence, Tuple, Callable, Optional, Union
import os
import time

from absl import flags
from absl import app
from absl import logging
import ase
import ase.data
import ase.io
import ase.visualize
import jax
import jax.numpy as jnp
import jraph
import flax
import numpy as np
import tqdm
import chex
import optax

import analyses.analysis as analysis
from symphony import datatypes
from symphony.data import input_pipeline
from symphony import models

FLAGS = flags.FLAGS


def append_predictions(
    pred: datatypes.Predictions,
    padded_fragment: datatypes.Fragments,
    max_num_atoms: int,
    num_target_positions: int,
    radial_cutoff: float,
    eps: float = 1e-4
) -> datatypes.Fragments:
    """Appends the predictions to a single fragment."""
    target_relative_positions = pred.globals.position_vectors[0]
    focus_indices = pred.globals.focus_indices[0]
    focus_positions = padded_fragment.nodes.positions[focus_indices]
    extra_positions = (target_relative_positions + focus_positions).reshape(-1, 3)
    extra_species = (pred.globals.target_species[0]).reshape(-1,)
    stop = pred.globals.stop

    filtered_positions = []
    position_mask = np.ones(len(extra_positions), dtype=bool)
    for i in range(len(extra_positions)):
        if not position_mask[i]:
            continue
        if eps > np.linalg.norm(extra_positions[i]) > 0:
            position_mask[i] = False
        else:
            filtered_positions.append(extra_positions[i])
    extra_positions = np.array(filtered_positions)
    extra_species = extra_species[position_mask]
    extra_len = min(len(extra_positions), max_num_atoms)

    new_positions = np.concatenate([fragment.nodes.positions, extra_positions[:extra_len]], axis=0)
    new_species = np.concatenate([fragment.nodes.species, extra_species[:extra_len]], axis=0)

    atomic_numbers = np.asarray([1, 6, 7, 8, 9])
    new_fragment = input_pipeline.ase_atoms_to_jraph_graph(
        atoms=ase.Atoms(numbers=atomic_numbers[new_species], positions=new_positions),
        atomic_numbers=atomic_numbers,
        radial_cutoff=radial_cutoff,
    )
    new_fragment = new_fragment._replace(globals=padded_fragment.globals)
    return stop, new_fragment


def generate_one_step(
    padded_fragment: datatypes.Fragments,
    stop: bool,
    rng: chex.PRNGKey,
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    max_num_atoms: int,
    max_targets: int,
    radial_cutoff: float,
) -> Tuple[
    Tuple[datatypes.Fragments, bool], Tuple[datatypes.Fragments, datatypes.Predictions]
]:
    """Generates the next fragment for a given seed."""
    pred = apply_fn(padded_fragment, rng)
    next_padded_fragment = append_predictions(pred, padded_fragment, max_num_atoms, max_targets, radial_cutoff)
    stop = pred.globals.stop[0] | stop
    return jax.lax.cond(
        stop,
        lambda: ((padded_fragment, True), (padded_fragment, pred)),
        lambda: ((next_padded_fragment, False), (next_padded_fragment, pred)),
    )


def generate_for_one_seed(
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    init_fragment: datatypes.Fragments,
    max_num_atoms: int,
    max_targets: int,
    cutoff: float,
    rng: chex.PRNGKey,
    return_intermediates: bool = False,
) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
    """Generates a single molecule for a given seed."""
    step_rngs = jax.random.split(rng, num=max_num_atoms)
    (final_padded_fragment, stop), (padded_fragments, preds) = jax.lax.scan(
        lambda args, rng: generate_one_step(*args, rng, apply_fn, max_num_atoms, max_targets, cutoff),
        (init_fragment, False),
        step_rngs,
    )
    if return_intermediates:
        return padded_fragments, preds
    else:
        return final_padded_fragment, stop


def generate_molecules(
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    params: optax.Params,
    molecules_outputdir: str,
    radial_cutoff: float,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    num_seeds: int,
    num_seeds_per_chunk: int,
    init_molecules: Sequence[Union[str, ase.Atoms]],
    max_num_atoms: int,
    avg_neighbors_per_atom: int,
    atomic_numbers: np.ndarray = np.array([1, 6, 7, 8, 9]),
    visualize: bool = False,
    visualizations_dir: Optional[str] = None,
    verbose: bool = True,
):
    """Generates molecules from a model."""

    if verbose:
        logging_fn = logging.info
    else:
        logging_fn = lambda *args: None

    # Create output directories.
    os.makedirs(molecules_outputdir, exist_ok=True)
    if visualize:
        assert visualizations_dir is not None
        os.makedirs(visualizations_dir, exist_ok=True)

    # Check that we can divide the seeds into chunks properly.
    if num_seeds % num_seeds_per_chunk != 0:
        raise ValueError(
            f"num_seeds ({num_seeds}) must be divisible by num_seeds_per_chunk ({num_seeds_per_chunk})"
        )

    # Create initial molecule, if provided.
    if isinstance(init_molecules, str):
        init_molecule, init_molecule_name = analysis.construct_molecule(init_molecules)
        logging_fn(
            f"Initial molecule: {init_molecule.get_chemical_formula()} with numbers {init_molecule.numbers} and positions {init_molecule.positions}"
        )
        init_molecules = [init_molecule] * num_seeds
        init_molecule_names = [init_molecule_name] * num_seeds
    else:
        assert len(init_molecules) == num_seeds
        init_molecule_names = [
            init_molecule.get_chemical_formula() for init_molecule in init_molecules
        ]

    if visualize:
        pass

    # Prepare initial fragments.
    init_fragments = [
        input_pipeline.ase_atoms_to_jraph_graph(
            init_molecule, atomic_numbers, radial_cutoff
        )
        for init_molecule in init_molecules
    ]
    init_fragments = [
        jraph.pad_with_graphs(
            init_fragment,
            n_node=(max_num_atoms + 1),
            n_edge=(max_num_atoms + 1) * avg_neighbors_per_atom,
            n_graph=2,
        )
        for init_fragment in init_fragments
    ]
    init_fragments = jax.tree_util.tree_map(lambda *val: np.stack(val), *init_fragments)
    init_fragments = jax.vmap(
        lambda init_fragment: jax.tree_util.tree_map(jnp.asarray, init_fragment)
    )(init_fragments)

    # Ensure params are frozen.
    params = flax.core.freeze(params)

    max_targets = init_fragments.globals.position_vectors[0].shape[0]

    @jax.jit
    def chunk_and_apply(
        init_fragments: datatypes.Fragments, rngs: chex.PRNGKey
    ) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
        """Chunks the seeds and applies the model sequentially over all chunks."""

        def apply_on_chunk(
            init_fragments_and_rngs: Tuple[datatypes.Fragments, chex.PRNGKey],
        ) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
            """Applies the model on a single chunk."""
            init_fragments, rngs = init_fragments_and_rngs
            assert len(init_fragments.n_node) == len(rngs)

            apply_fn_wrapped = lambda padded_fragment, rng: apply_fn(
                params,
                rng,
                padded_fragment,
                focus_and_atom_type_inverse_temperature,
                position_inverse_temperature,
            )
            generate_for_one_seed_fn = lambda rng, init_fragment: generate_for_one_seed(
                apply_fn_wrapped,
                init_fragment,
                max_num_atoms,
                max_targets,
                radial_cutoff,
                rng,
                return_intermediates=visualize,
            )
            return jax.vmap(generate_for_one_seed_fn)(rngs, init_fragments)

        # Chunk the seeds, apply the model, and unchunk the results.
        init_fragments, rngs = jax.tree_util.tree_map(
            lambda arr: jnp.reshape(
                arr,
                (num_seeds // num_seeds_per_chunk, num_seeds_per_chunk, *arr.shape[1:]),
            ),
            (init_fragments, rngs),
        )
        results = jax.lax.map(apply_on_chunk, (init_fragments, rngs))
        results = jax.tree_util.tree_map(lambda arr: arr.reshape((-1, *arr.shape[2:])), results)
        return results

    seeds = jnp.arange(num_seeds)
    rngs = jax.vmap(jax.random.PRNGKey)(seeds)

    # Compute compilation time.
    start_time = time.time()
    chunk_and_apply.lower(init_fragments, rngs).compile()
    compilation_time = time.time() - start_time
    logging_fn("Compilation time: %.2f s", compilation_time)

    # Generate molecules (and intermediate steps, if visualizing).
    if visualize:
        padded_fragments, preds = chunk_and_apply(init_fragments, rngs)
    else:
        final_padded_fragments, stops = chunk_and_apply(init_fragments, rngs)

    molecule_list = []
    for seed in tqdm.tqdm(seeds, desc="Visualizing molecules"):
        init_fragment = jax.tree_util.tree_map(lambda x: x[seed], init_fragments)
        init_molecule_name = init_molecule_names[seed]

        if visualize:
            # Get the padded fragment and predictions for this seed.
            padded_fragments_for_seed = jax.tree_util.tree_map(
                lambda x: x[seed], padded_fragments
            )
            preds_for_seed = jax.tree_util.tree_map(lambda x: x[seed], preds)

            figs = []
            for step in range(max_num_atoms):
                if step == 0:
                    padded_fragment = init_fragment
                else:
                    padded_fragment = jax.tree_util.tree_map(
                        lambda x: x[step - 1], padded_fragments_for_seed
                    )
                pred = jax.tree_util.tree_map(lambda x: x[step], preds_for_seed)

                # Save visualization of generation process.
                fragment = jraph.unpad_with_graphs(padded_fragment)
                pred = jraph.unpad_with_graphs(pred)
                fragment = fragment._replace(
                    globals=jax.tree_util.tree_map(
                        lambda x: np.squeeze(x, axis=0), fragment.globals
                    )
                )
                pred = pred._replace(
                    globals=jax.tree_util.tree_map(lambda x: np.squeeze(x, axis=0), pred.globals)
                )
                fig = analysis.visualize_predictions(pred, fragment)
                figs.append(fig)

                # This may be the final padded fragment.
                final_padded_fragment = padded_fragment

                # Check if we should stop.
                stop = pred.globals.stop
                if stop:
                    break

            # Save the visualizations of the generation process.
            for index, fig in enumerate(figs):
                # Update the title.
                fig.update_layout(
                    title=f"Predictions for Seed {seed}",
                    title_x=0.5,
                )

                # Save to file.
                outputfile = os.path.join(
                    visualizations_dir,
                    f"seed_{seed}_fragments_{index}.html",
                )
                fig.write_html(outputfile, include_plotlyjs="cdn")

        else:
            # We already have the final padded fragment.
            final_padded_fragment = jax.tree_util.tree_map(
                lambda x: x[seed], final_padded_fragments
            )
            stop = jax.tree_util.tree_map(lambda x: x[seed], stops)

        num_valid_nodes = final_padded_fragment.n_node[0]
        generated_molecule = ase.Atoms(
            positions=final_padded_fragment.nodes.positions[:num_valid_nodes],
            numbers=models.get_atomic_numbers(
                final_padded_fragment.nodes.species[:num_valid_nodes],
                atomic_numbers
            ),
        )
        if stop:
            logging_fn("Generated %s", generated_molecule.get_chemical_formula())
            outputfile = f"{init_molecule_name}_seed={seed}.xyz"
        else:
            logging_fn("STOP was not produced. Discarding...")
            outputfile = f"{init_molecule_name}_seed={seed}_no_stop.xyz"

        ase.io.write(os.path.join(molecules_outputdir, outputfile), generated_molecule)
        molecule_list.append(generated_molecule)

    return molecule_list, molecules_outputdir



def generate_molecules_from_workdir(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: Union[str, int],
    steps_for_weight_averaging: Optional[Sequence[int]],
    num_seeds: int,
    num_seeds_per_chunk: int,
    init_molecules: Sequence[Union[str, ase.Atoms]],
    max_num_atoms: int,
    avg_neighbors_per_atom: int,
    atomic_numbers: np.ndarray,
    visualize: bool = False,
    res_alpha: Optional[int] = None,
    res_beta: Optional[int] = None,
    verbose: bool = False,    
):
    """Generates molecules from a trained model at the given workdir."""

    # Load model.
    workdir = os.path.abspath(workdir)
    if steps_for_weight_averaging is not None:
        logging.info("Loading model averaged from steps %s", steps_for_weight_averaging)
        model, params, config = analysis.load_weighted_average_model_at_steps(
            workdir, steps_for_weight_averaging, run_in_evaluation_mode=True
        )
    else:
        model, params, config = analysis.load_model_at_step(
            workdir,
            step,
            run_in_evaluation_mode=True,
        )

    # Update resolution of sampling grid.
    config = config.unlock()
    if res_alpha is not None:
        logging.info(f"Setting res_alpha to {res_alpha}")
        config.target_position_predictor.res_alpha = res_alpha

    if res_beta is not None:
        logging.info(f"Setting res_beta to {res_beta}")
        config.target_position_predictor.res_beta = res_beta
    logging.info(config.to_dict())

    # Create output directories.
    name = analysis.name_from_workdir(workdir)
    molecules_outputdir = os.path.join(
        outputdir,
        name,
        f"fait={focus_and_atom_type_inverse_temperature}",
        f"pit={position_inverse_temperature}",
        f"step={step}",
    )
    molecules_outputdir += f"_res_alpha={config.target_position_predictor.res_alpha}"
    molecules_outputdir += f"_res_beta={config.target_position_predictor.res_beta}"
    molecules_outputdir += "/molecules"

    if visualize:
        visualizations_dir = os.path.join(
            outputdir,
            name,
            f"fait={focus_and_atom_type_inverse_temperature}",
            f"pit={position_inverse_temperature}",
            f"step={step}",
            "visualizations",
            "generated_molecules",
        )

    return generate_molecules(
            apply_fn=jax.jit(model.apply),
            params=params,
            molecules_outputdir=molecules_outputdir,
            radial_cutoff=config.radial_cutoff,
            focus_and_atom_type_inverse_temperature=focus_and_atom_type_inverse_temperature,
            position_inverse_temperature=position_inverse_temperature,
            num_seeds=num_seeds,
            num_seeds_per_chunk=num_seeds_per_chunk,
            init_molecules=init_molecules,
            max_num_atoms=max_num_atoms,
            avg_neighbors_per_atom=avg_neighbors_per_atom,
            atomic_numbers=atomic_numbers,
            visualize=visualize,
            visualizations_dir=visualizations_dir,
            verbose=verbose,
        )

def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    generate_molecules_from_workdir(
        FLAGS.workdir,
        FLAGS.outputdir,
        FLAGS.focus_and_atom_type_inverse_temperature,
        FLAGS.position_inverse_temperature,
        FLAGS.step,
        FLAGS.steps_for_weight_averaging,
        FLAGS.num_seeds,
        FLAGS.num_seeds_per_chunk,
        FLAGS.init,
        FLAGS.max_num_atoms,
        FLAGS.avg_neighbors_per_atom,
        FLAGS.visualize,
        FLAGS.res_alpha,
        FLAGS.res_beta,
        verbose=True,
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
    flags.DEFINE_string(
        "step",
        "best",
        "Step number to load model from. The default corresponds to the best model.",
    )
    flags.DEFINE_integer(
        "res_alpha",
        None,
        "Angular resolution of alpha.",
    )
    flags.DEFINE_integer(
        "res_beta",
        None,
        "Angular resolution of beta.",
    )
    flags.DEFINE_integer(
        "num_seeds",
        128,
        "Seeds to attempt to generate molecules from.",
    )
    flags.DEFINE_integer(
        "num_seeds_per_chunk",
        32,
        "Number of seeds evaluated in parallel. Reduce to avoid OOM errors.",
    )
    flags.DEFINE_string(
        "init",
        "C",
        "An initial molecular fragment to start the generation process from.",
    )
    flags.DEFINE_integer(
        "max_num_atoms",
        30,
        "Maximum number of atoms to generate per molecule.",
    )
    flags.DEFINE_integer(
        "avg_neighbors_per_atom",
        10,
        "Average number of neighbors per atom.",
    )
    flags.DEFINE_bool(
        "visualize",
        False,
        "Whether to visualize the generation process step-by-step.",
    )
    flags.DEFINE_list(
        "steps_for_weight_averaging",
        None,
        "Steps to average parameters over. If None, the model at the given step is used.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
