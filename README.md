# GMACE

### Instructions

Clone the repository:

```shell
git clone git@github.com:atomicarchitects/spherical-harmonic-net.git
cd spherical-harmonic-net
```

Are you using MIT SuperCloud? If so, follow the instructions in the section ["MIT SuperCloud Setup"](https://github.com/atomicarchitects/spherical-harmonic-net/edit/main/README.md#mit-supercloud-setup).
Otherwise, continue with ["Default Setup"](https://github.com/atomicarchitects/spherical-harmonic-net/edit/main/README.md#default-setup).

#### Default Setup
Create and activate a virtual environment:

```shell
python -m venv .venv && source .venv/bin/activate
```

Install pip dependencies with:

```shell
pip install --upgrade pip && pip install -r requirements.txt
```

Running the analysis script requires `openbabel==3.1.1`. This can be installed through conda.

Continue to ["Checking Installation"](https://github.com/atomicarchitects/spherical-harmonic-net/edit/main/README.md#checking-installation).

#### MIT SuperCloud Setup

```shell
module load anaconda/2023a
pip install -r supercloud_requirements.txt
```

Continue to ["Checking Installation"](https://github.com/atomicarchitects/spherical-harmonic-net/edit/main/README.md#checking-installation).

#### Checking Installation
Check that installation suceeded by running a short test:

```shell
python -m train_test
```

#### Start a Training Run 
Start training with a configuration defined
under `configs/`:

```shell
python -m main --workdir=./workdirs --config=configs/graphnet.py
```

#### Changing Hyperparameters

Since the configuration is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags),
you can override hyperparameters. For example, to change the number of training
steps, the batch size and the dataset:

```shell
python main.py --workdir=./workdirs --config=configs/graphnet.py \
--config.num_training_steps=10 --config.batch_size=50
```

For more extensive changes, you can directly edit the configuration files,
and even add your own.

#### Generating Molecules

```python generate_molecules.py --workdir=PATH/TO/MODEL --outputdir=. --visualize=True```

 * `--workdir` Path to the model directory.
 * `--outputdir` Location to save the generated molecules.
 * `--step` Step of the model to use for generation. The default of -1 will use the best model.
 * `--beta` Temperature parameter for molecule generation.
 * `--seeds` List of seeds to use for molecule generation.
 * `--init` An initial molecule fragment to start generation from; "C" by default.
 * `--visualize` If True, the generated molecules are visualized in plotly, and the resulting graphs are saved as .html files.

#### Analysis Scripts

`qm9_filter_generated.py` is adapted from [G-SchNet](https://github.com/atomistic-machine-learning/G-SchNet). From the main directory, run
```python -m analyses.qm9_filter_generated PATH/TO/MOLECULES --model_path=PATH/TO/MODEL --data_path=PATH/TO/DATA --print_file```

 * `mol_path` Path to the generated molecules, stored in .mol_dict format (see analyses/utility_classes.py for the Molecule class).
 * `--data_path` Path to a database containing the training/validation/test data.
 * `--model_path` Path to the model used to generate the molecules.
 * `--print_file` If saving the console output to a file, add this flag and redirect the console output to a separate file when calling the script. For example:
 ```python -m analyses.qm9_filter_generated PATH/TO/MOLECULES --model_path=PATH/TO/MODEL --data_path=PATH/TO/DATA --print_file >> ./results.txt```

This script checks the valency constraints (e.g. every hydrogen atom should have exactly one bond), the connectedness (i.e. all atoms in a molecule should be connected to each other via a path over bonds), and removes duplicates. The remaining valid structures are stored in an sqlite database with ASE (in the same folder as the generated molecules), along with a pickle file containing:
 * losses from each stage of model training (i.e. evaluated against the training, validation, and test sets)
 * overall statistics for the generated molecules (e.g. the proportion of atoms/molecules with valid valences, the number of duplicated molecules, the number of novel molecules)
 * statistics for each generated molecule (e.g. the number of rings of certain sizes, the number of single, double, and triple bonds, the index of the matching training/test data molecule etc. for each molecule).

Note: This script uses molecular fingerprints and canonical smiles representations to identify duplicates, which means that different structures corresponding to the same canonical smiles string are tagged as duplicates and removed in the process. Add `--filters valence disconnected` to the call in order to keep identified duplicates in the created database.