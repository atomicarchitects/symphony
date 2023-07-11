"""Creates an ASE database containing training/validation/test data."""

import ml_collections
import os
from typing import Optional, Sequence

from absl import flags
from absl import app

import ase
import jax

import analysis
import input_pipeline
import utility_classes


FLAGS = flags.FLAGS


def dataset_as_database(
    config: ml_collections.ConfigDict,
    dataset: str,
    outputdir: str,
    qm9dir: Optional[str] = None,
) -> None:
    """Converts the dataset to a ASE database.
    Args:
        config (ml_collections.ConfigDict)
        dataset (str): should be 'train', 'val', 'test', or 'all'
        dbpath (str): path to save the ASE database to
        root_dir (str, optional): root dir for qm9
    """

    if qm9dir is None:
        qm9dir = config.root_dir
    _, _, molecules = input_pipeline.get_raw_datasets(
        rng=jax.random.PRNGKey(config.rng_seed), config=config, root_dir=qm9dir
    )
    compressor = utility_classes.ConnectivityCompressor()
    counter = 0
    to_convert = ["train", "val", "test"] if dataset == "all" else [dataset]
    dbpath = os.path.join(outputdir, f"qm9_{dataset}.db")
    with ase.db.connect(dbpath) as conn:
        for s in to_convert:
            for atoms in molecules[s]:
                # instantiate utility_classes.Molecule object
                mol = utility_classes.Molecule(atoms.positions, atoms.numbers)
                # get connectivity matrix (detecting bond orders with Open Babel)
                con_mat = mol.get_connectivity()
                conn.write(atoms, data={"con_mat": compressor.compress(con_mat)})
                counter += 1


def main(unused_argv: Sequence[str]) -> None:
    del unused_argv

    qm9dir = os.path.abspath(FLAGS.qm9dir)
    workdir = FLAGS.workdir
    outputdir = FLAGS.outputdir
    datasets = FLAGS.datasets

    config, _, _, _ = analysis.load_from_workdir(workdir)

    dataset_as_database(config, datasets, outputdir, qm9dir)


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
    flags.DEFINE_string(
        "qm9dir",
        os.path.join(os.getcwd(), "qm9_data"),
        "Directory where QM9 data is stored.",
    )
    flags.mark_flags_as_required(["workdir"])
    app.run(main)
