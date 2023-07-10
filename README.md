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
python -m symphony --workdir=./workdirs --config=configs/graphnet.py
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
