"""Creates an ASE database containing training/validation/test data."""

from typing import List, Sequence

import os

from absl import flags
from absl import app

import analysis


FLAGS = flags.FLAGS

def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    qm9_dir = os.path.abspath(FLAGS.qm9_dir)
    workdir = FLAGS.workdir
    outputdir = FLAGS.outputdir
    datasets = FLAGS.datasets

    config, _, _, _ = analysis.load_from_workdir(workdir)

    analysis.dataset_as_database(config, datasets, outputdir, qm9_dir)


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for model.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd()),
        "Directory where resulting database should be saved.",
    )
    flags.DEFINE_string(
        "datasets",
        "all",
        "The data split(s) that should be saved in the database (train, val, test, all)",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
