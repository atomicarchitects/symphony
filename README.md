# Symphony

### Instructions

Clone the repository:

```shell
git clone git@github.com:atomicarchitects/symphony.git
cd symphony
```

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

#### Checking Installation
Check that installation suceeded by running a test:

```shell
python -m tests.train_test
```

#### Start a Training Run 
Start training with a configuration defined
under `configs/`:

```shell
python -m symphony --workdir=./workdirs --config=configs/qm9/nequip.py
```

#### Changing Hyperparameters

Since the configuration is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags),
you can override hyperparameters. For example, to change the number of training
steps and the learning rate:

```shell
python main.py --workdir=./workdirs --config=configs/qm9/nequip.py \
--config.num_training_steps=10 --config.learning_rate=1e-3
```

For more extensive changes, you can directly edit the configuration files,
and even add your own.
