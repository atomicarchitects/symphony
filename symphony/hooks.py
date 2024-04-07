from dataclasses import dataclass
from typing import Sequence, Callable, Dict, Any
import os
import time
import tempfile
import pickle

import flax
import chex
from absl import logging
import flax.struct
from rdkit import Chem
import wandb
from clu import metric_writers, checkpoint


from symphony import train, train_state
from symphony import graphics
from analyses import metrics, generate_molecules


def add_prefix_to_keys(result: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Adds a prefix to the keys of a dict, returning a new dict."""
    return {f"{prefix}/{key}": val for key, val in result.items()}


def plot_molecules_in_wandb(
    molecules: Sequence[Chem.Mol],
    step: int,
    num_to_plot: int = 25,
    **plot_kwargs,
):
    """Plots molecules in the Weights & Biases UI."""

    if wandb.run is None:
        logging.info("No Weights & Biases run found. Skipping plotting of molecules.")
        return

    # Limit the number of molecules to plot.
    molecules = molecules[:num_to_plot]

    # Plot and save the view to a temporary HTML file.
    view = graphics.plot_molecules_with_py3Dmol(molecules, **plot_kwargs)
    temp_html_path = os.path.join(tempfile.gettempdir(), f"{wandb.run.name}.html")
    view.write_html(temp_html_path)

    # Log the HTML file to Weights & Biases.
    wandb.run.log({"samples": wandb.Html(open(temp_html_path))}, step=step)

    # Delete the temporary HTML file, after a short delay.
    time.sleep(1)
    os.remove(temp_html_path)


@dataclass
class GenerateMoleculesHook:
    workdir: str
    writer: metric_writers.SummaryWriter
    eval_apply_fn: Callable
    focus_and_atom_type_inverse_temperature: float
    position_inverse_temperature: float
    res_alpha: int
    res_beta: int
    nn_cutoff: float
    num_seeds: int
    num_seeds_per_chunk: int
    init_molecules: str
    max_num_atoms: int

    def __call__(self, state: train_state.TrainState) -> None:
        molecules_outputdir = os.path.join(
            self.workdir,
            "molecules",
            f"fait={self.focus_and_atom_type_inverse_temperature}",
            f"pit={self.position_inverse_temperature}",
            f"res_alpha={self.res_alpha}",
            f"res_beta={self.res_beta}",
            f"nn_cutoff={self.nn_cutoff}",
            f"step={state.get_step()}",
        )

        molecules_ase, _ = generate_molecules.generate_molecules(
            apply_fn=self.eval_apply_fn,
            params=flax.jax_utils.unreplicate(state.params),
            molecules_outputdir=molecules_outputdir,
            nn_cutoff=self.nn_cutoff,
            focus_and_atom_type_inverse_temperature=self.focus_and_atom_type_inverse_temperature,
            position_inverse_temperature=self.position_inverse_temperature,
            num_seeds=self.num_seeds,
            num_seeds_per_chunk=self.num_seeds_per_chunk,
            init_molecules=self.init_molecules,
            max_num_atoms=self.max_num_atoms,
            visualize=False,
            verbose=False,
        )
        logging.info(
            "Generated and saved %d molecules at %s",
            len(molecules_ase),
            molecules_outputdir,
        )

        # Convert to RDKit molecules.
        molecules = metrics.ase_to_rdkit_molecules(molecules_ase)

        # Compute metrics.
        validity = metrics.compute_validity(molecules)
        uniqueness = metrics.compute_uniqueness(molecules)

        # Write metrics out.
        self.writer.write_scalars(
            state.get_step(),
            {
                "validity": validity,
                "uniqueness": uniqueness,
            },
        )
        self.writer.flush()

        # Plot molecules.
        plot_molecules_in_wandb(molecules, state.get_step())


@dataclass
class LogTrainMetricsHook:
    writer: metric_writers.SummaryWriter
    is_empty: bool = True

    def __call__(self, state: train_state.TrainState) -> None:
        train_metrics = flax.jax_utils.unreplicate(state.train_metrics)

        # If the metrics are not empty, log them.
        # Once logged, reset the metrics, and mark as empty.
        if not self.is_empty:
            self.writer.write_scalars(
                int(state.get_step()),
                add_prefix_to_keys(train_metrics.compute(), "train"),
            )
            state = state.replace(
                train_metrics=flax.jax_utils.replicate(train.Metrics.empty())
            )
            self.is_empty = True


@dataclass
class EvaluateModelHook:
    evaluate_model_fn: Callable
    writer: metric_writers.SummaryWriter
    update_state: bool = True

    def __call__(
        self, state: train_state.TrainState, rng: chex.PRNGKey
    ) -> train_state.TrainState:
        # Evaluate the model.
        eval_metrics = self.evaluate_model_fn(
            state,
            rng,
        )

        # Compute and write metrics.
        for split in eval_metrics:
            eval_metrics[split] = eval_metrics[split].compute()
            self.writer.write_scalars(
                state.get_step(), add_prefix_to_keys(eval_metrics[split], split)
            )
        self.writer.flush()

        if not self.update_state:
            return state

        # Note best state seen so far.
        # Best state is defined as the state with the lowest validation loss.
        try:
            min_val_loss = state.metrics_for_best_params["val_eval"]["total_loss"]
        except (AttributeError, KeyError):
            logging.info("No best state found yet.")
            min_val_loss = float("inf")

        if eval_metrics["val_eval"]["total_loss"] < min_val_loss:
            state = state.replace(
                best_params=state.params,
                metrics_for_best_params=flax.jax_utils.replicate(eval_metrics),
                step_for_best_params=state.step,
            )
            logging.info("New best state found at step %d.", state.get_step())

        return state


@dataclass
class CheckpointHook:
    checkpoint_dir: str
    max_to_keep: int

    def __init__(self, checkpoint_dir: str, max_to_keep: int):
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.ckpt = checkpoint.Checkpoint(
            self.checkpoint_dir, max_to_keep=self.max_to_keep
        )

    def restore_or_initialize(
        self, state: train_state.TrainState
    ) -> train_state.TrainState:
        restored = self.ckpt.restore_or_initialize(
            {
                "state": state,
            }
        )
        state = restored["state"]
        return state

    def __call__(self, state: train_state.TrainState) -> Any:
        # Save the current and best params.
        with open(
            os.path.join(self.checkpoint_dir, f"params_{state.get_step()}.pkl"), "wb"
        ) as f:
            pickle.dump(flax.jax_utils.unreplicate(state.params), f)

        with open(os.path.join(self.checkpoint_dir, "params_best.pkl"), "wb") as f:
            pickle.dump(flax.jax_utils.unreplicate(state.best_params), f)

        # Save the whole training state.
        self.ckpt.save(
            {
                "state": flax.jax_utils.unreplicate(state),
            }
        )
