from typing import Dict, Tuple, Sequence, Union
import concurrent.futures
import os
import time

from absl import app
from absl import flags
from absl import logging
import ase
import tqdm
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from analyses import analysis
from symphony import datatypes
from symphony.data import input_pipeline
from symphony import models

FLAGS = flags.FLAGS

def append_predictions_to_fragment(
    fragment: datatypes.Fragments, pred: datatypes.Predictions, valid: bool, radial_cutoff: float,
) -> Tuple[int, datatypes.Fragments]:
    """Appends the predictions to a single fragment."""
    # If padding graph, just return the fragment.
    if not valid:
        return True, fragment

    target_relative_positions = pred.globals.position_vectors[0]
    focus_index = pred.globals.focus_indices[0]
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


def create_batch_iterator(all_fragments: Sequence[datatypes.Fragments], stopped: Sequence[bool], padding_budget: Dict[str, int]):
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


def generate_molecules(
    workdir: str,
    outputdir: str,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: str,
    num_seeds: int,
    init_molecules: Sequence[Union[str, ase.Atoms]],
    max_num_atoms: int,
    avg_atoms_per_graph: int,
    avg_neighbors_per_atom: int,
    num_seeds_per_chunk: int,
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

    # Load model.
    rng = jax.random.PRNGKey(0)
    name = analysis.name_from_workdir(workdir)
    model, params, config = analysis.load_model_at_step(
        workdir, step, run_in_evaluation_mode=True
    )
    params = jax.device_put(params)

    @jax.jit
    def apply_fn(batch, apply_rng):
        preds = model.apply(
            params,
            apply_rng,
            batch,
            focus_and_atom_type_inverse_temperature,
            position_inverse_temperature,
        )
        return batch, preds

    # Log config.
    logging_fn(config.to_dict())

    # Create output directories.
    molecules_outputdir = os.path.join(
        outputdir,
        name,
        f"fait={focus_and_atom_type_inverse_temperature}",
        f"pit={position_inverse_temperature}",
        f"step={step}",
        "molecules",
    )
    os.makedirs(molecules_outputdir, exist_ok=True)

    all_fragments = [
        input_pipeline.ase_atoms_to_jraph_graph(
            init_molecule, models.ATOMIC_NUMBERS, config.radial_cutoff
        )
        for init_molecule in init_molecules
    ]
    stopped = [False] * len(all_fragments)

    num_atoms_per_chunk = avg_atoms_per_graph * (num_seeds_per_chunk + 1)
    padding_budget = dict(
        n_node=num_atoms_per_chunk,
        n_edge=num_atoms_per_chunk * avg_neighbors_per_atom * 2,
        n_graph=(num_seeds_per_chunk + 1),
    )

    # Start timer.
    start_time = time.time()

    # Create a ThreadPoolExecutor for parallel execution on the CPU
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

        # Keep track of progress.
        pbar = tqdm.tqdm(desc="Generating molecules")

        while True:
            # Process all batches.
            indices, gpu_futures = [], []
            for fragment_indices, fragments in create_batch_iterator(all_fragments, stopped, padding_budget):
                apply_rng, rng = jax.random.split(rng)
                
                # Predict on this batch.
                gpu_future = executor.submit(apply_fn, fragments, apply_rng)
                gpu_futures.append(gpu_future)
                indices.extend(fragment_indices)

            # Nothing left to process.
            if len(gpu_futures) == 0:
                break

            # Wait for the GPU computation to complete.
            gpu_results = [future.result() for future in gpu_futures]

            # Process the results on the CPU.
            cpu_futures = []
            for fragments, preds in gpu_results:
                
                # Bring back to CPU.
                fragments = jax.tree_util.tree_map(np.asarray, fragments)
                preds = jax.tree_util.tree_map(np.asarray, preds)
                valids = jraph.get_graph_padding_mask(fragments)

                for fragment, pred, valid in zip(jraph.unbatch(fragments), jraph.unbatch(preds), valids):
                    cpu_future = executor.submit(append_predictions_to_fragment, fragment, pred, valid, config.radial_cutoff)
                    cpu_futures.append(cpu_future)

            # Wait for all the CPU tasks to complete.
            cpu_results = [future.result() for future in cpu_futures]

            # Update the input data with the CPU results.
            assert len(indices) == len(cpu_results)
            for index, (stop, new_fragment) in zip(indices, cpu_results):
                if index is None:
                    continue

                stopped[index] |= stop
                stopped[index] |= (len(new_fragment.nodes.species) > max_num_atoms)
                
                if not stopped[index]:
                    all_fragments[index] = new_fragment

            # Update progress bar.
            pbar.update()

    # Stop timer.
    elapsed_time = time.time() - start_time

    # Log generation time.
    logging_fn(
        f"Generated {len(all_fragments)} molecules in {elapsed_time} seconds."
    )
    logging_fn(
        f"Average time per molecule: {elapsed_time / len(all_fragments)} seconds."
    )

    generated_molecules_ase = []
    for index, (stop, fragment) in enumerate(zip(stopped, all_fragments)):
        init_molecule_name = init_molecule_names[index]
        generated_molecule_ase = ase.Atoms(
            symbols=models.utils.get_atomic_numbers(fragment.nodes.species),
            positions=fragment.nodes.positions,
        )

        if stop:
            logging_fn("Generated %s", generated_molecule_ase.get_chemical_formula())
            output_file = f"{init_molecule_name}_seed={index}.xyz"
        else:
            logging_fn("STOP was not produced ...")
            output_file = f"{init_molecule_name}_seed={index}_no_stop.xyz"

        generated_molecule_ase.write(os.path.join(molecules_outputdir, output_file))
        generated_molecules_ase.append(generated_molecule_ase)

    return generated_molecules_ase


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    workdir = os.path.abspath(FLAGS.workdir)
    generate_molecules(
        workdir,
        FLAGS.outputdir,
        FLAGS.focus_and_atom_type_inverse_temperature,
        FLAGS.position_inverse_temperature,
        FLAGS.step,
        FLAGS.num_seeds,
        FLAGS.init,
        FLAGS.max_num_atoms,
        FLAGS.avg_atoms_per_graph,
        FLAGS.avg_neighbors_per_atom,
        FLAGS.num_seeds_per_chunk,
        FLAGS.verbose,
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
        10,
        "Maximum number of atoms to generate per molecule.",
    )
    flags.DEFINE_integer(
        "avg_atoms_per_graph",
        20,
        "Average number of atoms per graph.",
    )
    flags.DEFINE_integer(
        "avg_neighbors_per_atom",
        10,
        "Average number of neighbors per atom.",
    )
    flags.DEFINE_integer(
        "num_seeds_per_chunk",
        64,
        "Number of seeds to process in parallel.",
    )
    flags.DEFINE_bool(
        "verbose",
        True,
        "Whether to print verbose output.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
