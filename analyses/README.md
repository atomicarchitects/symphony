# Analysis Scripts

We provide a set of scripts to analyze the performance of the models along various aspects.

Run these scripts from the top-level directory of the repository.
For more details on supported flags,
pass `--help` to the script.

## Plot Performance of Different Models

```bash
python -m analyses.generate_plots --basedir="${BASEDIR}"/v5                     
```

`${BASEDIR}` is the directory where all of the workdirs are found.
You can change the version number ("v5") depending on which models you want to plot.


## Plot Sample Complexity Curves of Different Models

A sample complexity curve is the 
```bash
python -m analyses.generate_plots --basedir="${BASEDIR}"/extras/sample_complexity                   
```

## Generate Molecules from a Model

```bash
python -m analyses.generate_molecules --workdir="${WORKDIR}" --beta="${BETA}"
```

`beta` here refers to the inverse temperature for generation,
and is 1 by default.


## Relax Structures of Generated Molecules

First, generate molecules from your model with the script above. Then:

```bash
python -m analyses.relax_structures --workdir="${WORKDIR}" --beta="${BETA}"
```

## Plot Model Predictions on Atom Removal from a Molecule

We can evaluate the model by visualizing its prediction on removing a single atom from a molecule.

```bash
python -m analyses.visualize_atom_removals --workdir="${WORKDIR}" --molecule=...
```

We provide some example molecules in `analyses/molecules/downloaded/` to analyse in this manner.
