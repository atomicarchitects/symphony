"""Relaxes structures using ASE."""

from typing import Tuple, Sequence, Optional

from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT
import ase.io
import os
import pandas as pd
from absl import app
from absl import flags
from absl import logging

import sys

sys.path.append("..")

import analyses.analysis as analysis

# Try to import ORCA and Psi4 calculators
AVAILABLE_CALCULATORS = []
try:
    from ase.calculators.orca import ORCA

    ORCA()
    logging.info("Found ORCA calculator")
    AVAILABLE_CALCULATORS.append("ORCA")
except:
    pass
try:
    from ase.calculators.psi4 import Psi4

    Psi4()
    logging.info("Found Psi4 calculator")
    AVAILABLE_CALCULATORS.append("PSI4")
except:
    pass

FLAGS = flags.FLAGS


def relax_all_structures(moldir: str, outputdir: str) -> pd.DataFrame:
    """Relaxes all structures in a directory."""

    results = []
    for atoms_file in os.listdir(moldir):
        if atoms_file.endswith(".xyz"):
            input_file = os.path.join(moldir, atoms_file)
            output_file = os.path.join(
                outputdir, atoms_file.removesuffix(".xyz") + ".traj"
            )

            molecule = ase.io.read(input_file)
            init_energy, final_energy, delta_energy = relax_structure(
                molecule, output_file, label=atoms_file.removesuffix(".xyz")
            )
            formula = molecule.get_chemical_formula(mode="hill")
            results.append([formula, init_energy, final_energy, delta_energy])

    results = pd.DataFrame(
        results, columns=["formula", "init_energy", "final_energy", "delta_energy"]
    )
    results = results.astype(
        {
            "formula": str,
            "init_energy": float,
            "final_energy": float,
            "delta_energy": float,
        }
    )
    return results


def relax_structure(
    molecule: ase.Atoms,
    output_file: str,
    calculator: Optional[str] = None,
    label: Optional[str] = None,
) -> Tuple[float, float, float]:
    """Relaxes a structure using ASE, saving the trajectory to a file."""
    if label is None:
        label = molecule.get_chemical_formula(mode="hill")

    if calculator not in AVAILABLE_CALCULATORS:
        if calculator is not None:
            logging.info(
                "Calculator %s not available, using first available calculator from %s.",
                calculator,
                AVAILABLE_CALCULATORS,
            )
        calculator = AVAILABLE_CALCULATORS[0]

    if calculator == "ORCA":
        molecule.calc = ORCA(
            label=label,
            orcasimpleinput="B3LYP 6-31G(2df,2p)",
            orcablock="%pal nprocs 8 end",
        )
    elif calculator == "PSI4":
        molecule.calc = Psi4(
            method="b3lyp", basis="6-31g_2df_p_", memory="4000MB", threads=1
        )

    dyn = BFGS(molecule, trajectory=output_file)
    dyn.run(fmax=0.01)

    traj = ase.io.Trajectory(output_file)
    init_energy = traj[0].get_potential_energy()
    final_energy = traj[-1].get_potential_energy()
    delta_energy = init_energy - final_energy

    return init_energy, final_energy, delta_energy


def main(unused_argv: Sequence[str]):
    del unused_argv

    workdir = FLAGS.workdir
    name = analysis.name_from_workdir(workdir)
    moldir = os.path.join(FLAGS.outputdir, "molecules", name, f"beta={FLAGS.beta}")

    outputdir = os.path.join(FLAGS.outputdir, "relaxations", name, f"beta={FLAGS.beta}")
    os.makedirs(outputdir, exist_ok=True)

    results = relax_all_structures(moldir, outputdir)
    logging.info("Relaxation results:")
    logging.info(results)


if __name__ == "__main__":
    flags.DEFINE_string("workdir", None, "Workdir for model.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses"),
        "Directory where molecules should be saved.",
    )
    flags.DEFINE_float("beta", 1.0, "Inverse temperature value for sampling.")

    flags.mark_flags_as_required(["workdir"])
    app.run(main)
