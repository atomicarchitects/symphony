"""Generates molecules from a trained model."""

from typing import Sequence, Tuple, Callable, Optional, Union, Dict
import os
import time

from absl import flags
from absl import app
from absl import logging
import ase
import ase.data
#from ase.db import connect
import ase.io
import ase.visualize
import biotite.structure as struc
from biotite.structure.io import pdb
import jax
import jax.numpy as jnp
import jraph
import flax
import numpy as np
import tqdm
import chex
import optax
import time

import analyses.analysis as analysis
from symphony import datatypes, models
from symphony.data import input_pipeline
from symphony.data.datasets import qm9, tmqm, cath

FLAGS = flags.FLAGS


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


def append_predictions(
    pred: datatypes.Predictions,
    padded_fragment: datatypes.Fragments,
    radial_cutoff: float,
) -> datatypes.Fragments:
    """Appends the predictions to the padded fragment."""
    # Update the positions of the first dummy node.
    positions = padded_fragment.nodes.positions
    num_valid_nodes = padded_fragment.n_node[0]
    num_nodes = padded_fragment.nodes.positions.shape[0]
    num_edges = padded_fragment.receivers.shape[0]
    focus = pred.globals.focus_indices[0]
    focus_position = positions[focus]
    target_position = pred.globals.position_vectors[0] + focus_position
    new_positions = positions.at[num_valid_nodes].set(target_position)

    # Update the species of the first dummy node.
    species = padded_fragment.nodes.species
    target_species = pred.globals.target_species[0]
    new_species = species.at[num_valid_nodes].set(target_species)

    # Compute the distance matrix to select the edges.
    distance_matrix = jnp.linalg.norm(
        new_positions[None, :, :] - new_positions[:, None, :], axis=-1
    )
    node_indices = jnp.arange(num_nodes)

    # Avoid self-edges.
    valid_edges = (distance_matrix > 0) & (distance_matrix < radial_cutoff)
    valid_edges = (
        valid_edges
        & (node_indices[None, :] <= num_valid_nodes)
        & (node_indices[:, None] <= num_valid_nodes)
    )
    senders, receivers = jnp.nonzero(
        valid_edges, size=num_edges, fill_value=-1
    )
    num_valid_edges = jnp.sum(valid_edges)
    num_valid_nodes += 1

    return padded_fragment._replace(
        nodes=padded_fragment.nodes._replace(
            positions=new_positions,
            species=new_species,
        ),
        n_node=jnp.asarray([num_valid_nodes, num_nodes - num_valid_nodes]),
        n_edge=jnp.asarray([num_valid_edges, num_edges - num_valid_edges]),
        senders=senders,
        receivers=receivers,
    )


def generate_one_step(
    padded_fragment: datatypes.Fragments,
    stop: bool,
    rng: chex.PRNGKey,
    apply_fn: Callable[[datatypes.Fragments, chex.PRNGKey], datatypes.Predictions],
    radial_cutoff: float,
) -> Tuple[
    Tuple[datatypes.Fragments, bool], Tuple[datatypes.Fragments, datatypes.Predictions]
]:
    """Generates the next fragment for a given seed."""
    pred = apply_fn(padded_fragment, rng)
    next_padded_fragment = append_predictions(pred, padded_fragment, radial_cutoff)
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
    cutoff: float,
    rng: chex.PRNGKey,
    return_intermediates: bool = False,
) -> Tuple[datatypes.Fragments, datatypes.Predictions]:
    """Generates a single molecule for a given seed."""
    step_rngs = jax.random.split(rng, num=max_num_atoms)
    (final_padded_fragment, stop), (padded_fragments, preds) = jax.lax.scan(
        lambda args, rng: generate_one_step(*args, rng, apply_fn, cutoff),
        (init_fragment, False),
        step_rngs,
    )
    if return_intermediates:
        return padded_fragments, preds
    else:
        return final_padded_fragment, stop


def bfs_ordering(positions, species):
    struct = datatypes.Structures(
        nodes=datatypes.NodesInfo(
            positions=positions, species=species
        ),
        edges=None,
        receivers=None,
        senders=None,
        globals=None,
        n_node=jnp.asarray([len(species)]),
        n_edge=None,
    )
    # add edges between nearest neighbors
    struct = input_pipeline.infer_edges_with_radial_cutoff_on_positions(
        struct, 1.7
    )
    visited = np.zeros_like(species, dtype=bool)
    start_ndx = 0
    for i in range(len(species)):
        if species[i] == 25:  # "X" the starting N atom
            start_ndx = i
            if len(struct.receivers[struct.senders == i]) == 1:
                break
    queue = [start_ndx]
    indices = []
    while queue:
        node = queue.pop(0)
        if visited[node]:
            continue
        visited[node] = True
        indices.append(node)
        neighbors = struct.receivers[struct.senders == node]
        queue.extend(neighbors)
    indices = np.array(indices)
    return positions[indices], species[indices]


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
    """Generates molecules from a model."""

    if verbose:
        logging_fn = logging.info
    else:
        logging_fn = lambda *args: None

    # Create output directories.
    os.makedirs(molecules_outputdir, exist_ok=True)

    # Set parameters based on the dataset.
    if "qm9" in dataset:
        max_num_atoms = 35
        avg_nodes_per_graph = 35
        avg_edges_per_graph = 350
        species_to_atomic_numbers = qm9.QM9Dataset.species_to_atomic_numbers()
        atoms_to_species = qm9.QM9Dataset.atoms_to_species()
    elif dataset == "tmqm":
        max_num_atoms = 60
        avg_nodes_per_graph = 50
        avg_edges_per_graph = 500
        species_to_atomic_numbers = tmqm.TMQMDataset.species_to_atomic_numbers()
        atoms_to_species = tmqm.TMQMDataset.atoms_to_species()
    elif dataset == "platonic_solids":
        max_num_atoms = 35
        avg_nodes_per_graph = 35
        avg_edges_per_graph = 175
        species_to_atomic_numbers = {0: 1}
        atoms_to_species = {1: 0}
    elif dataset == "cath":
        max_num_atoms = 512
        avg_nodes_per_graph = 512
        avg_edges_per_graph = 512 * 5
        species_to_atomic_numbers = cath.CATHDataset.species_to_atomic_numbers()
        atoms_to_species = cath.CATHDataset.atoms_to_species()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Check that we can divide the seeds into chunks properly.
    if num_seeds % num_seeds_per_chunk != 0:
        raise ValueError(
            f"num_seeds ({num_seeds}) must be divisible by num_seeds_per_chunk ({num_seeds_per_chunk})"
        )

    # Create initial molecule, if provided.
    if isinstance(init_molecules, str):
        if dataset == "cath":
            init_molecule, init_molecule_name = analysis.construct_backbone(init_molecules)
            init_positions = init_molecule.coord
            init_atomic_symbols = init_molecule.atom_name
            # replace CB with their respective amino acids
            cb_mask = init_atomic_symbols == "CB"
            n_atoms = len(init_atomic_symbols)
            cb_indices = np.arange(n_atoms)[cb_mask]
            cb_residues = struc.get_residue_positions(init_molecule, cb_indices)
            init_atomic_symbols[cb_mask] = init_molecule.res_name[cb_residues]
        else:
            init_molecule, init_molecule_name = analysis.construct_molecule(init_molecules)
            init_positions = init_molecule.positions
            init_atomic_symbols = init_molecule.symbols
        logging_fn(
            f"Initial molecule: {init_molecule_name} with atoms {init_atomic_symbols} and positions {init_positions}"
        )
        init_molecules = [
            input_pipeline.to_jraph_graph(
                init_positions, init_atomic_symbols, atoms_to_species, radial_cutoff,
            )
        ] * num_seeds
        init_molecule_names = [init_molecule_name] * num_seeds
    elif isinstance(init_molecules[0], ase.Atoms):
        assert len(init_molecules) == num_seeds
        init_molecule_names = [
            init_molecule.get_chemical_formula() for init_molecule in init_molecules
        ]
        init_molecules = [
            input_pipeline.to_jraph_graph(
                init_molecule.positions, init_molecule.symbols, atoms_to_species, radial_cutoff,
            )
            for init_molecule in init_molecules
        ]
    else:
        init_molecule_names = [f"mol_{i}" for i in range(len(init_molecules))]

    # Prepare initial fragments.
    padding_budget = estimate_padding_budget(
        init_molecules[:10], num_seeds_per_chunk,
        avg_nodes_per_graph, avg_edges_per_graph,
        padding_mode=padding_mode,
    )
    init_fragments = [
        jraph.pad_with_graphs(
            init_fragment,
            n_node=(max_num_atoms + 1),
            n_edge=(max_num_atoms + 1) * avg_edges_per_graph / avg_nodes_per_graph,
            n_graph=2,
        )
        for init_fragment in init_molecules
    ]
    init_fragments = jax.tree_util.tree_map(lambda *val: np.stack(val), *init_fragments)
    init_fragments = jax.vmap(
        lambda init_fragment: jax.tree_util.tree_map(jnp.asarray, init_fragment)
    )(init_fragments)

    # Ensure params are frozen.
    params = flax.core.freeze(params)

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
                radial_cutoff,
                rng,
                return_intermediates=False,
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

    seeds = jnp.arange(start_seed, num_seeds+start_seed)
    rngs = jax.vmap(jax.random.PRNGKey)(seeds)

    # Compute compilation time.
    start_time = time.time()
    chunk_and_apply.lower(init_fragments, rngs).compile()
    compilation_time = time.time() - start_time
    logging_fn("Compilation time: %.2f s", compilation_time)

    final_padded_fragments, stops = chunk_and_apply(init_fragments, rngs)
    # n_node shape = [num_seeds, 2]
    n_nodes = final_padded_fragments.n_node[:, 0].astype(int)
    molecule_list = []
    if dataset == "cath":
        filetype = "pdb"
    else:
        filetype = "xyz"
    for i, seed in tqdm.tqdm(enumerate(seeds), desc="Processing molecules"):
        init_molecule_name = init_molecule_names[i]
        positions = final_padded_fragments.nodes.positions[i, :n_nodes[i], :]
        species = final_padded_fragments.nodes.species[i, :n_nodes[i]]
        if stops[i]:
            logging_fn("Generated %s", generated_molecule.get_chemical_formula())
            outputfile = f"{init_molecule_name}_seed={seed}.xyz"
        else:
            logging_fn("STOP was not produced. Discarding...")
            outputfile = f"{init_molecule_name}_seed={seed}_no_stop.{filetype}"
        # write protein backbones to pdb file
        if dataset == "cath":
            def pdb_line(atom, atomnum, resname, resnum, x, y, z):
                return f"ATOM  {atomnum:5}  {atom:3} {resname:3} A{resnum:4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            if not len(species):
                logging_fn("No residues found in molecule. Discarding...")
                continue
            # in order for biotite to properly process the pdb file, atoms need to be properly assigned to residues
            positions, species = bfs_ordering(positions, species)
            # prepare pdb file
            lines = []
            residue_start_ndx = 0
            curr_residue = ""
            species_names = cath.CATHDataset.get_species()
            residue_species = []
            residue_ct = 1
            for ndx in range(n_nodes[i]):
                if species[ndx] < 22:  # amino acid
                    curr_residue = species_names[species[ndx]]
                    residue_species.append("CB")
                elif species[ndx] >= 24 and ndx != 0:  # N
                    for k in range(residue_start_ndx, ndx):
                        lines.append(pdb_line(
                            residue_species[k - residue_start_ndx],
                            k + 1,
                            curr_residue,
                            residue_ct,
                            *positions[k]
                        ))
                    residue_start_ndx = ndx
                    residue_ct += 1
                    residue_species = ["N"]
                else:
                    residue_species.append(species_names[species[ndx]])
            if len(residue_species) > 0:
                for k in range(residue_start_ndx, n_nodes[i]):
                    lines.append(pdb_line(
                        residue_species[k - residue_start_ndx],
                        k + 1,
                        curr_residue,
                        residue_ct,
                        *positions[k]
                    ))
            with open(os.path.join(molecules_outputdir, outputfile), "w") as f:
                f.write("\n".join(lines))
            pdb_file = pdb.PDBFile.read(os.path.join(molecules_outputdir, outputfile))
            generated_molecule = pdb.get_structure(pdb_file)
        # for regular molecules it's much simpler
        else:
            dict_species = jnp.array(list(species_to_atomic_numbers.keys()))
            dict_numbers = jnp.array(list(species_to_atomic_numbers.values()))
            generated_molecule = ase.Atoms(
                positions=positions,
                numbers=jax.vmap(
                    lambda x: dict_numbers[jnp.where(dict_species == x, size=1)[0]]
                )(species).flatten(),
            )
            ase.io.write(os.path.join(molecules_outputdir, outputfile), generated_molecule)
        molecule_list.append(generated_molecule)

    return molecule_list



def generate_molecules_from_workdir(
    workdir: str,
    outputdir: str,
    radial_cutoff: float,
    focus_and_atom_type_inverse_temperature: float,
    position_inverse_temperature: float,
    step: Union[str, int],
    steps_for_weight_averaging: Optional[Sequence[int]],
    start_seed: int,
    num_seeds: int,
    num_seeds_per_chunk: int,
    init_molecules: Sequence[Union[str, ase.Atoms]],
    dataset: str,
    padding_mode: str,
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
    molecules_outputdir += f"_res_alpha={config.generation.res_alpha}"
    molecules_outputdir += f"_res_beta={config.generation.res_beta}"
    molecules_outputdir += "/molecules"

    return generate_molecules(
            apply_fn=jax.jit(model.apply),
            params=params,
            molecules_outputdir=molecules_outputdir,
            radial_cutoff=radial_cutoff,
            focus_and_atom_type_inverse_temperature=focus_and_atom_type_inverse_temperature,
            position_inverse_temperature=position_inverse_temperature,
            start_seed=start_seed,
            num_seeds=num_seeds,
            num_seeds_per_chunk=num_seeds_per_chunk,
            init_molecules=init_molecules,
            dataset=dataset,
            padding_mode=padding_mode,
            verbose=verbose,
        )

def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    generate_molecules_from_workdir(
        FLAGS.workdir,
        FLAGS.outputdir,
        FLAGS.radial_cutoff,
        FLAGS.focus_and_atom_type_inverse_temperature,
        FLAGS.position_inverse_temperature,
        FLAGS.step,
        FLAGS.steps_for_weight_averaging,
        FLAGS.start_seed,
        FLAGS.num_seeds,
        FLAGS.num_seeds_per_chunk,
        FLAGS.init,
        FLAGS.dataset,
        FLAGS.padding_mode,
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
        "radial_cutoff",
        5.0,
        "Radial cutoff for edge finding"
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
        "start_seed",
        0,
        "Initial seed."
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
    flags.DEFINE_list(
        "steps_for_weight_averaging",
        None,
        "Steps to average parameters over. If None, the model at the given step is used.",
    )
    flags.DEFINE_string(
        "dataset",
        "qm9",
        "Dataset from which to generate molecules.",
    )
    flags.DEFINE_string(
        "padding_mode",
        "dynamic",
        "How to determine molecule padding.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
