# Analysis Scripts

We provide a set of scripts to analyze the performance of the models along various aspects.

Run these scripts from the top-level directory of the repository.
For more details on supported flags,
pass `--help` to the script.

A workdir (`WORKDIR` below) is where all of the data for a particular experiment is stored. Some scripts analyse a single experiment, while others analyse multiple.

## Plot Performance of Different Models

```bash
python -m analyses.generate_plots --basedir="${BASEDIR}"/v5                     
```

`BASEDIR` is the directory where all of the workdirs are found.
You can change the version number (`v5`) depending on which models you want to plot.


## Plot Sample Complexity Curves of Different Models

A sample complexity curve is a plot of the model's performance as a function of the number of training samples.

```bash
python -m analyses.generate_plots --basedir="${BASEDIR}"/extras/sample_complexity                   
```

## Generate Molecules from a Model

```bash
python -m analyses.generate_molecules --workdir="${WORKDIR}" --beta="${BETA}"
```

Selected flags:
 * `--workdir` Path to the model directory.
 * `--outputdir` Location to save the generated molecules; a folder will be created inside the current directory by default.
 * `--step` Step of the model to use for generation. The default of -1 will use the best model.
 * `--beta` Inverse temperature for generation; 1 by default.
 * `--init` An initial molecule fragment to start generation from; "C" by default.


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

## Analyze Generated Molecules

`analyze_generated.py` is adapted from [G-SchNet](https://github.com/atomistic-machine-learning/G-SchNet).

```bash
python -m analyses.analyze_generated ${MOL_PATH} --model_path="${MODEL_PATH}" --data_path="${DATA_PATH}" --init="${INIT}"
```

Selected flags:
 * `mol_path` Path to the generated molecules, stored in .mol_dict format (see analyses/utility_classes.py for the Molecule class).
 * `--data_path` Path to a database containing the training/validation/test data.
 * `--model_path` Path to the model used to generate the molecules.
 * `--init` The initial molecule fragment used to generate the molecules.

This script checks the valency constraints (e.g. every hydrogen atom should have exactly one bond), the connectedness (i.e. all atoms in a molecule should be connected to each other via a path over bonds), and removes duplicates. The remaining valid structures are stored in an sqlite database with ASE (in the same folder as the generated molecules), along with a pickle file containing:
 * losses from each stage of model training (i.e. evaluated against the training, validation, and test sets)
 * overall statistics for the generated molecules (e.g. the proportion of atoms/molecules with valid valences, the number of duplicated molecules, the number of novel molecules)
 * statistics for each generated molecule (e.g. the number of rings of certain sizes, the number of single, double, and triple bonds, the index of the matching training/test data molecule etc. for each molecule).
