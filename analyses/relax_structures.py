from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT
import ase.io
import numpy as np
import os

from absl import app
from absl import flags

def main(argv):
    del argv
    outfile = 'analyses/H2O.traj'

    d = 0.9575
    t = np.pi / 180 * 104.51
    water = ase.Atoms('H2O',
                positions=[(d, 0, 0),
                            (d * np.cos(t), d * np.sin(t), 0),
                            (0, 0, 0)],
                calculator=EMT())
    
    dyn = BFGS(water, trajectory=outfile)
    dyn.run(fmax=0.001)

    traj = ase.io.Trajectory(outfile)
    init_energy = traj[0].get_potential_energy()
    final_energy = traj[-1].get_potential_energy()
    delta_energy = final_energy - init_energy

    print()
    print("Relaxation complete.")
    print(f"* E_init: {init_energy}")
    print(f"* E_final: {final_energy}")
    print(f"* Delta E: {delta_energy}")


if __name__ == "__main__":
    # flags.DEFINE_string("workdir", None, "Workdir.")
    # flags.DEFINE_string(
    #     "outputdir",
    #     os.path.join(os.getcwd(), "analyses", "outputs", "relax_structures"),
    #     "Directory where plots should be saved.",
    # )

    # flags.mark_flags_as_required(["basedir"])
    app.run(main)
