from typing import Dict, Tuple, Sequence, Union, Callable
import concurrent.futures
import os
import time

from absl import app
from absl import flags
from absl import logging
import ase
import chex
import jax.numpy as jnp
import optax
import jax
import jraph
import numpy as np

from analyses import analysis
from symphony import datatypes
from symphony.data import input_pipeline
from symphony.data.datasets import qm9, qm9_single, tmqm
from symphony import models


FLAGS = flags.FLAGS


def append_predictions_to_fragment(
    fragment: datatypes.Fragments,
    pred: datatypes.Predictions,
    valid: bool,
    radial_cutoff: float,
) -> Tuple[int, datatypes.Fragments]:
    """Appends the predictions to a single fragment."""
    # If padding graph, just return the fragment.
    if not valid:
        return True, fragment

    target_relative_positions = pred.globals.position_vectors[0]
    focus_index = pred.globals.focus_indices[0]  # TODO uh. well. oop
    focus_position = fragment.nodes.positions[focus_index]
    extra_position = target_relative_positions + focus_position
    extra_species = pred.globals.target_species[focus_index]
    stop = pred.globals.stop

    new_positions = np.concatenate([fragment.nodes.positions, [extra_position]], axis=0)
    new_species = np.concatenate([fragment.nodes.species, [extra_species]], axis=0)

    atomic_numbers = np.asarray([1, 6, 7, 8, 9])
    new_fragment = input_pipeline.ase_atoms_to_jraph_graph(
        atoms=ase.Atoms(numbers=atomic_numbers[new_species], positions=new_positions),
        atomic_numbers=atomic_numbers,
        radial_cutoff=radial_cutoff,
    )
    new_fragment = new_fragment._replace(globals=fragment.globals)
    return stop, new_fragment


def create_batch_iterator(
    all_fragments: Sequence[datatypes.Fragments],
    stopped: Sequence[bool],
    padding_budget: Dict[str, int],
):
    """Creates a iterator over batches."""
    assert len(all_fragments) == len(stopped)

    indices, batch = [], []
    for index, data in enumerate(all_fragments):
        if stopped[index]:
            continue

        indices.append(index)
        batch.append(data)

        if len(batch) == padding_budget["n_graph"] - 1:
            indices = indices + [None] * (padding_budget["n_graph"] - len(batch))
            batch = jraph.batch_np(batch)
            batch = jraph.pad_with_graphs(batch, **padding_budget)
            yield indices, batch
            indices, batch = [], []

    if len(batch) > 0:
        indices = indices + [None] * (padding_budget["n_graph"] - len(batch))
        batch = jraph.pad_with_graphs(jraph.batch_np(batch), **padding_budget)
        yield indices, batch


def estimate_padding_budget(
    all_fragments: Sequence[datatypes.Fragments],
    num_seeds_per_chunk: int, 
    avg_nodes_per_graph: int,
    avg_edges_per_graph: int,
    padding_mode: str,
):
    """Estimates the padding budget for a batch."""

    def round_to_nearest_multiple_of_64(x):
        return int(np.ceil(x / 64) * 64)

    if padding_mode == "fixed":
        avg_nodes_per_graph = 50
        avg_edges_per_graph = 1000
    elif padding_mode == "dynamic":
        avg_nodes_per_graph = sum(
            fragment.n_node.sum() for fragment in all_fragments
        ) / len(all_fragments)
        avg_edges_per_graph = sum(
            fragment.n_edge.sum() for fragment in all_fragments
        ) / len(all_fragments)
    else:
        raise ValueError(f"Unknown padding mode: {padding_mode}")

    avg_nodes_per_graph = max(avg_nodes_per_graph, 1)
    avg_edges_per_graph = max(avg_edges_per_graph, 1)
    padding_budget = dict(
        n_node=round_to_nearest_multiple_of_64(
            num_seeds_per_chunk * avg_nodes_per_graph * 1.5
        ),
        n_edge=round_to_nearest_multiple_of_64(
            num_seeds_per_chunk * avg_edges_per_graph * 1.5
        ),
        n_graph=num_seeds_per_chunk,
    )
    return padding_budget


def generate_molecules_from_workdir(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: str,
    num_seeds: int,
    init_molecules: Sequence[Union[str, ase.Atoms]],
    num_seeds_per_chunk: int,
    dataset: str,
    padding_mode: str,
    verbose: bool = True,
):
    """Generates molecules from a trained model at the given workdir."""
    if verbose:
        logging_fn = logging.info
    else:
        logging_fn = lambda *args: None

    logging_fn("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging_fn("JAX local devices: %r", jax.local_devices())
    logging_fn("CUDA_VISIBLE_DEVICES: %r", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # Load model.
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )
    params = jax.device_put(params)

    # Log config.
    logging_fn(config.to_dict())

    # Set output directory.
    molecules_outputdir = os.path.join(
        outputdir,
        name,
        f"fait={focus_and_atom_type_inverse_temperature}",
        f"pit={position_inverse_temperature}",
        f"step={step}",
        "molecules",
    )

    # Generate molecules.
    return generate_molecules(
        apply_fn=model.apply,
        params=params,
        molecules_outputdir=molecules_outputdir,
        radial_cutoff=config.radial_cutoff,
        focus_and_atom_type_inverse_temperature=focus_and_atom_type_inverse_temperature,
        position_inverse_temperature=position_inverse_temperature,
        num_seeds=num_seeds,
        init_molecules=init_molecules,
        num_seeds_per_chunk=num_seeds_per_chunk,
        dataset=dataset,
        padding_mode=padding_mode,
        verbose=verbose,
    )


def generate_molecules(
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    params: optax.Params,
    molecules_outputdir: str,
    radial_cutoff: float,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    start_seed: int,
    num_seeds: int,
    num_seeds_per_chunk: int,
    init_molecules: Sequence[Union[str, ase.Atoms]],
    dataset: str,
    padding_mode: str,
    verbose: bool = False,
):
    if verbose:
        logging_fn = logging.info
    else:
        logging_fn = lambda *args: None

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

    # Set parameters based on the dataset.
    if "qm9" in dataset:
        max_num_atoms = 35
        avg_nodes_per_graph = 35
        avg_edges_per_graph = 350
        atomic_numbers = qm9.QM9Dataset.get_atomic_numbers()
    elif dataset == "tmqm":
        max_num_atoms = 60
        avg_nodes_per_graph = 50
        avg_edges_per_graph = 500
        atomic_numbers = tmqm.TMQMDataset.get_atomic_numbers()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    @jax.jit
    def apply_fn_wrapped(batch, apply_rng):
        preds = apply_fn(
            params,
            apply_rng,
            batch,
            focus_and_atom_type_inverse_temperature,
            position_inverse_temperature,
        )
        return batch, preds

    # Create output directories.
    os.makedirs(molecules_outputdir, exist_ok=True)

    # Create initial fragments.
    all_fragments = [
        input_pipeline.ase_atoms_to_jraph_graph(
            init_molecule, atomic_numbers, radial_cutoff
        )
        for init_molecule in init_molecules
    ]
    stopped = [False] * len(all_fragments)
    iteration_count = 0
    rng = jax.random.PRNGKey(0)
 
    # Start timer.
    start_time = time.time()

    # Create a ThreadPoolExecutor for parallel execution on the CPU
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:

        while True:
            # Keep track of start time for this iteration.
            iteration_start_time = time.time()

            # Update padding budget.
            padding_budget = estimate_padding_budget(
                all_fragments[:10], num_seeds_per_chunk,
                avg_nodes_per_graph, avg_edges_per_graph,
                padding_mode=padding_mode,
            )
            logging_fn("Padding budget for this iteration: %r", padding_budget)

            # Process all batches.
            indices, gpu_futures = [], []
            for fragment_indices, fragments in create_batch_iterator(
                all_fragments, stopped, padding_budget
            ):
                apply_rng, rng = jax.random.split(rng)

                # Predict on this batch.
                gpu_future = executor.submit(apply_fn_wrapped, fragments, apply_rng)
                gpu_futures.append(gpu_future)
                indices.extend(fragment_indices)

            # Nothing left to process.
            if len(gpu_futures) == 0:
                break

            # Wait for the GPU computation to complete.
            gpu_results = [future.result() for future in gpu_futures]

            # Process the results on the CPU.
            cpu_results = []
            for fragments, preds in gpu_results:

                # Bring back to CPU.
                fragments = jax.tree_util.tree_map(np.asarray, fragments)
                preds = jax.tree_util.tree_map(np.asarray, preds)
                valids = jraph.get_graph_padding_mask(fragments)

                for fragment, pred, valid in zip(
                    jraph.unbatch(fragments), jraph.unbatch(preds), valids
                ):
                    # cpu_future = executor.submit(append_predictions_to_fragment, fragment, pred, valid, config.radial_cutoff)
                    # cpu_futures.append(cpu_future)

                    stop, new_fragment = append_predictions_to_fragment(
                        fragment, pred, valid, radial_cutoff
                    )
                    cpu_results.append((stop, new_fragment))

            # # Wait for all the CPU tasks to complete.
            # cpu_results = [future.result() for future in cpu_futures]

            # Update the input data with the CPU results.
            assert len(indices) == len(cpu_results)
            for index, (stop, new_fragment) in zip(indices, cpu_results):
                if index is None:
                    continue

                stopped[index] |= stop
                stopped[index] |= len(new_fragment.nodes.species) > max_num_atoms

                if not stopped[index]:
                    all_fragments[index] = new_fragment

            # Update iteration count.
            iteration_count += 1

            # Log iteration time.
            iteration_elapsed_time = time.time() - iteration_start_time
            logging_fn(
                "Iteration %d took %0.2f seconds.",
                iteration_count,
                iteration_elapsed_time,
            )

    # Stop timer.
    elapsed_time = time.time() - start_time

    # Log generation time.
    logging_fn(
        f"Generated {len(all_fragments)} molecules in {elapsed_time:.2f} seconds."
    )
    logging_fn(
        f"Average time per molecule: {elapsed_time / len(all_fragments):.2f} seconds."
    )

    generated_molecules_ase = []
    for seed, stop, fragment in zip(range(start_seed, start_seed+num_seeds), stopped, all_fragments):
        init_molecule_name = init_molecule_names[seed]
        generated_molecule_ase = ase.Atoms(
            symbols=models.get_atomic_numbers(fragment.nodes.species, atomic_numbers),
            positions=fragment.nodes.positions,
        )

        if stop:
            output_file = f"{init_molecule_name}_seed={seed}.xyz"
        else:
            output_file = f"{init_molecule_name}_seed={seed}_no_stop.xyz"

        generated_molecule_ase.write(os.path.join(molecules_outputdir, output_file))
        generated_molecules_ase.append(generated_molecule_ase)

    return generated_molecules_ase


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    generate_molecules_from_workdir(
        workdir=workdir,
        outputdir=FLAGS.outputdir,
        focus_and_atom_type_inverse_temperature=FLAGS.focus_and_atom_type_inverse_temperature,
        position_inverse_temperature=FLAGS.position_inverse_temperature,
        step=FLAGS.step,
        num_seeds=FLAGS.num_seeds,
        init_molecules=FLAGS.init,
        max_num_atoms=FLAGS.max_num_atoms,
        avg_nodes_per_graph=FLAGS.avg_nodes_per_graph,
        avg_edges_per_graph=FLAGS.avg_edges_per_graph,
        num_seeds_per_chunk=FLAGS.num_seeds_per_chunk,
        dataset=FLAGS.dataset,
        padding_mode=FLAGS.padding_mode,
    )


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
        "num_seeds",
        1000,
        "Seeds to attempt to generate molecules from.",
    )
    flags.DEFINE_string(
        "init",
        "C",
        "An initial molecular fragment to start the generation process from.",
    )
    flags.DEFINE_integer(
        "max_num_atoms",
        50,
        "Maximum number of atoms to generate per molecule.",
    )
    flags.DEFINE_integer(
        "avg_nodes_per_graph",
        50,
        "Average number of atoms per graph.",
    )
    flags.DEFINE_integer(
        "avg_edges_per_graph",
        500,
        "Average number of neighbors per atom.",
    )
    flags.DEFINE_integer(
        "num_seeds_per_chunk",
        8,
        "Number of seeds to process in parallel.",
    )
    flags.DEFINE_string(
        "padding_mode",
        "fixed",
        "Kind of padding to use.",
    )
    flags.DEFINE_bool(
        "verbose",
        True,
        "Whether to print verbose output.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
