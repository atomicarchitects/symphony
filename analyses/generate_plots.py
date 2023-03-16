"""Creates plots to analyze trained models."""

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
sys.path.append('..')

import analyses.analysis as analysis


FLAGS = flags.FLAGS

flags.DEFINE_string("basedir", None, "Directory where all workdirs are stored.")
flags.DEFINE_string("outputdir", os.path.join(os.getcwd(), "analyses", "outputs"), "Directory where plots should be saved.")
flags.DEFINE_string("model", None, "Model to analyze.")
flags.DEFINE_string("metric", "total_loss", "Metric to plot.")


def get_hyperparameters(config):
    """Returns the hyperparameters extracted from the config."""
    if "num_interactions" in config:
        num_interactions = config.num_interactions
    else:
        num_interactions = config.n_interactions

    max_l = config.max_ell

    if "num_channels" in config:
        num_channels = config.num_channels
    else:
        num_channels = config.n_atom_basis
        assert num_channels == config.n_filters
    
    return num_interactions, max_l, num_channels


def get_title_for_model(model: str) -> str:
    """Returns the title for the given model."""
    if model == "e3schnet":
        return "E3SchNet"
    elif model == "mace":
        return "MACE"
    return model.title()


def create_plot(df: pd.DataFrame, metric: str, title: str, filename: str):
    """Creates a boxplot for the given metric."""
    # Set style.
    sns.set_theme(style="darkgrid")

    # Scatterplot.
    fig, axs = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)
    fig.suptitle(title)

    for ax, num_interactions in zip(axs, [1, 2]):
        # Choose the subset of data based on the number of interactions.
        df_subset = df[df["num_interactions"] == num_interactions]

        # Scatterplot.
        ax = sns.scatterplot(data=df_subset, x="max_l", y=metric, edgecolor="gray", ax=ax,
                             hue="num_channels", size="num_channels", sizes=(50, 150))
        
        # Customizing different axes.
        if num_interactions == 1:
            ax.legend().remove()
            ax.set_ylabel(" ".join(ax.get_ylabel().split("_")).title())

        if num_interactions == 2:
            ax.legend(title="Number of Channels", loc="center left",
                      bbox_to_anchor=(1.04, 0.5), borderaxespad=0, fancybox=True, shadow=True)
            for ha in ax.legend_.legendHandles:
                ha.set_edgecolor("gray")

            ax.set_ylabel("")

        # Labels and titles.
        ax.set_title(f"{num_interactions} Interactions")
        ax.set_xlabel(" ".join(ax.get_xlabel().split("_")).title())
        ax.set_xticks(np.arange(df["max_l"].min(), df["max_l"].max() + 1))

        # Add jitter to the points.
        np.random.seed(0)
        dots = ax.collections[0]
        offsets = dots.get_offsets()
        jittered_offsets = np.stack([offsets[:, 0] + np.random.uniform(-0.1, 0.1, size=offsets[:, 0].shape), offsets[:, 1]], axis=1)
        dots.set_offsets(jittered_offsets)

    # Save plot.
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    basedir = os.path.abspath(FLAGS.basedir)
    metric = FLAGS.metric
    model = FLAGS.model
    outputdir = os.path.abspath(FLAGS.outputdir)

    validation_data = []
    test_data = []

    for config_file_path in glob.glob(os.path.join(basedir, "**", model, "**", "*.yml"), recursive=True):
        workdir = os.path.dirname(config_file_path)

        config, metrics_for_best_state = analysis.load_metrics_from_workdir(workdir)
        num_interactions, max_l, num_channels = get_hyperparameters(config)

        validation_metric = metrics_for_best_state["val"][metric]
        validation_data.append([num_interactions, max_l, num_channels, validation_metric])

        test_metric = metrics_for_best_state["test"][metric]
        test_data.append([num_interactions, max_l, num_channels, test_metric])

    validation_data = np.array(validation_data)
    validation_df = pd.DataFrame(validation_data, columns=["num_interactions", "max_l", "num_channels", f"validation_{metric}"])
    validation_df = validation_df.astype({"num_interactions": int, "max_l": int, "num_channels": int, f"validation_{metric}": float})
    create_plot(validation_df, f"validation_{metric}", get_title_for_model(model) + " on Validation Set", os.path.join(outputdir, f"{model}_validation_{metric}.png"))
    print(validation_df)
    test_data = np.array(test_data)
    test_df = pd.DataFrame(test_data, columns=["num_interactions", "max_l", "num_channels", f"test_{metric}"])
    test_df = test_df.astype({"num_interactions": int, "max_l": int, "num_channels": int, f"test_{metric}": float})
    create_plot(test_df, f"test_{metric}", get_title_for_model(model) + " on Test Set", os.path.join(outputdir, f"{model}_test_{metric}.png"))


if __name__ == "__main__":
    flags.mark_flags_as_required(["basedir", "model"])
    app.run(main)
