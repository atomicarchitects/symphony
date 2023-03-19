"""Creates plots to analyze trained models."""

from typing import Sequence, Dict

import ml_collections
from absl import app
from absl import flags
from absl import logging
import os
import glob
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.append("..")

import analyses.analysis as analysis


FLAGS = flags.FLAGS

flags.DEFINE_string("basedir", None, "Directory where all workdirs are stored.")
flags.DEFINE_string(
    "outputdir",
    os.path.join(os.getcwd(), "analyses", "outputs"),
    "Directory where plots should be saved.",
)
flags.DEFINE_string("model", None, "Model to analyze.")


def get_title_for_model(model: str) -> str:
    """Returns the title for the given model."""
    if model == "e3schnet":
        return "E3SchNet"
    elif model == "mace":
        return "MACE"
    return model.title()


def get_title_for_split(split):
    """Returns the title for the given split."""
    if split == "test":
        return "Test"
    elif split == "val":
        return "Validation"
    return split.title()


def plot_performance_for_parameters(
    model: str, metrics: Sequence[str], results: Dict[str, pd.DataFrame], outputdir: str
) -> None:
    """Creates a line plot for each metric as a function of number of parameters."""

    def plot_metric(metric: str, split: str):
        # Set style.
        sns.set_theme(style="darkgrid")

        # One figure for each value of num_interactions.
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
        fig.suptitle(get_title_for_model(model) + " on " + get_title_for_split(split) + " Set")

        for ax, num_interactions in zip(
            axs, results[split]["num_interactions"].drop_duplicates().sort_values()
        ):
            # Choose the subset of data based on the number of interactions.
            df = results[split]
            df_subset = df[df["num_interactions"] == num_interactions]

            # Lineplot.
            sns.lineplot(
                data=df_subset,
                x="num_params",
                y=metric,
                hue="max_l",
                style="max_l",
                markersize=10,
                markers=True,
                dashes=True,
                ax=ax,
            )

            # Customizing different axes.
            if num_interactions == 1:
                ax.legend().remove()
                ax.set_ylabel(" ".join(ax.get_ylabel().split("_")).title())

            if num_interactions == 2:
                ax.legend(
                    title="Max L",
                    loc="center left",
                    bbox_to_anchor=(1.04, 0.5),
                    borderaxespad=0,
                    fancybox=True,
                    shadow=False,
                )
                ax.set_ylabel("")

            # Axes limits.
            min_y = min(results[split][metric].min() for split in results)
            max_y = max(results[split][metric].max() for split in results)
            ax.set_ylim(min_y - 0.2, max_y + 0.2)

            # Labels and titles.
            ax.set_xlabel("Number of Parameters")
            ax.set_title(f"{num_interactions} Interactions")

            # Add jitter to the points.
            np.random.seed(0)
            dots = ax.collections[0]
            offsets = dots.get_offsets()
            jittered_offsets = np.stack(
                [
                    offsets[:, 0]
                    + np.random.uniform(-0.1, 0.1, size=offsets[:, 0].shape),
                    offsets[:, 1],
                ],
                axis=1,
            )
            dots.set_offsets(jittered_offsets)

        # Save figure.
        os.makedirs(os.path.join(outputdir, "num_params"), exist_ok=True)
        outputfile = os.path.join(
            outputdir, "num_params", f"{model}_{split}_{metric}_params.png"
        )
        plt.savefig(outputfile, bbox_inches="tight")
        plt.close()

    for split in results:
        for metric in metrics:
            plot_metric(metric, split)


def plot_performance_for_max_ell(
    model: str, metrics: Sequence[str], results: Dict[str, pd.DataFrame], outputdir: str
) -> None:
    """Creates a scatter plot for each metric, grouped by max_ell."""

    def plot_metric(metric: str, split: str):
        # Set style.
        sns.set_theme(style="darkgrid")

        # One figure for each value of num_interactions.
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
        fig.suptitle(get_title_for_model(model) + " on " + get_title_for_split(split) + " Set")

        for ax, num_interactions in zip(
            axs, results[split]["num_interactions"].drop_duplicates().sort_values()
        ):
            # Choose the subset of data based on the number of interactions.
            df = results[split]
            df_subset = df[df["num_interactions"] == num_interactions]

            # Scatterplot.
            ax = sns.scatterplot(
                data=df_subset,
                x="max_l",
                y=metric,
                hue="num_channels",
                size="num_channels",
                sizes=(100, 200),
                ax=ax,
            )

            # Customizing different axes.
            if num_interactions == 1:
                ax.legend().remove()
                ax.set_ylabel(" ".join(ax.get_ylabel().split("_")).title())

            if num_interactions == 2:
                ax.legend(
                    title="Number of Channels",
                    loc="center left",
                    bbox_to_anchor=(1.04, 0.5),
                    borderaxespad=0,
                    fancybox=True,
                    shadow=False,
                )
                ax.set_ylabel("")

            # Axes limits.
            min_y = min(results[split][metric].min() for split in results)
            max_y = max(results[split][metric].max() for split in results)
            ax.set_ylim(min_y - 0.2, max_y + 0.2)

            # Labels and titles.
            ax.set_title(f"{num_interactions} Interactions")
            ax.set_xlabel("Max L")
            ax.set_xticks(np.arange(df["max_l"].min(), df["max_l"].max() + 1))

            # Add jitter to the points.
            np.random.seed(0)
            dots = ax.collections[0]
            offsets = dots.get_offsets()
            jittered_offsets = np.stack(
                [
                    offsets[:, 0]
                    + np.random.uniform(-0.1, 0.1, size=offsets[:, 0].shape),
                    offsets[:, 1],
                ],
                axis=1,
            )
            dots.set_offsets(jittered_offsets)

        # Save plot.
        os.makedirs(os.path.join(outputdir, "max_ell"), exist_ok=True)
        outputfile = os.path.join(outputdir, "max_ell", f"{model}_{split}_{metric}_max_ell.png")
        plt.savefig(outputfile, bbox_inches="tight")
        plt.close()

    for split in results:
        for metric in metrics:
            plot_metric(metric, split)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    basedir = os.path.abspath(FLAGS.basedir)
    model = FLAGS.model
    outputdir = os.path.abspath(FLAGS.outputdir)
    metrics = ["total_loss", "position_loss", "focus_loss", "atom_type_loss"]

    # Get results.
    results = analysis.get_results_as_dataframe(model, metrics, basedir)

    # Make plots.
    plot_performance_for_max_ell(model, metrics, results, outputdir)
    plot_performance_for_parameters(model, metrics, results, outputdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["basedir", "model"])
    app.run(main)
