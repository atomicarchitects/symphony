"""Relaxes structures using ASE."""

from typing import Tuple, Sequence

from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.calculators.psi4 import Psi4
import ase.io
import os
import pandas as pd
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


def relax_all_structures(moldir: str, outputdir: str) -> pd.DataFrame:
    """Relaxes all structures in a directory."""

    results = []
    for atoms_file in os.listdir(moldir):
        if atoms_file.endswith(".xyz"):
            input_file = os.path.join(moldir, atoms_file)
            output_file = os.path.join(outputdir, atoms_file.removesuffix(".xyz") + ".traj")

            molecule = ase.io.read(input_file)
            init_energy, final_energy, delta_energy = relax_structure(molecule, output_file)
            formula = molecule.get_chemical_formula(mode='hill')
            results.append([formula, init_energy, final_energy, delta_energy])
    
    results = pd.DataFrame(
        results,
        columns=["formula", "init_energy", "final_energy", "delta_energy"]
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


def relax_structure(molecule: ase.Atoms, output_file: str) -> Tuple[float, float, float]:
    """Relaxes a structure using ASE, saving the trajectory to a file."""
    
    molecule.calc = Psi4(method='b3lyp', basis='6-311g_d_p_', memory=1000, threads=1)
    dyn = BFGS(molecule, trajectory=output_file)
    dyn.run(fmax=0.001)

    traj = ase.io.Trajectory(output_file)
    init_energy = traj[0].get_potential_energy()
    final_energy = traj[-1].get_potential_energy()
    delta_energy = init_energy - final_energy

    return init_energy, final_energy, delta_energy


def main(unused_argv: Sequence[str]):
    del unused_argv

    moldir = FLAGS.moldir
    workdir = moldir[moldir.find("molecules") + len("molecules/"):]
    outputdir = os.path.join(FLAGS.outputdir, workdir)
    os.makedirs(outputdir, exist_ok=True)

    results = relax_all_structures(moldir, outputdir)
    logging.info("Relaxation results:")
    logging.info(results)


if __name__ == "__main__":
    flags.DEFINE_string("moldir", None, "Directory of generated molecules.")
    flags.DEFINE_string(
        "outputdir",
        os.path.join(os.getcwd(), "analyses", "relaxations"),
        "Directory where output relaxation trajectories should be saved.",
    )

    flags.mark_flags_as_required(["moldir"])
    app.run(main)
