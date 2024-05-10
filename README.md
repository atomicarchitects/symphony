## Symphony: Symmetry-Equivariant Point-Centered Spherical Harmonics for Periodic Structure Generation

### Instructions

Clone the repository:

```shell
git clone git@github.com:atomicarchitects/symphony.git
cd symphony
git checkout iclr_2024_final
```

Since this repository is actively being developed, we recommend using the `iclr_2024_final` branch for the most stable version of the code.

#### Default Setup
Create and activate a virtual environment:

```shell
python -m venv .venv && source .venv/bin/activate
```

Install pip dependencies with:

```shell
pip install --upgrade pip && pip install -r requirements.txt
```

For GPU support, install JAX with CUDA support afterwards:
```shell
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Some of the analysis scripts require `openbabel==3.1.1`.
This can be installed through conda.


#### Checking Installation
Check that installation suceeded by running a short test:

```shell
python -m tests.train_test
```

#### Start a Training Run 
Start training with a configuration defined
under `configs/`:

```shell
python -m symphony \
    --config configs/qm9/e3schnet_and_nequip.py \
    --workdir ./workdirs
```

The `--workdir` flag specifies the directory where the
model checkpoints, logs, and other artifacts will be saved.

#### Changing Hyperparameters

Since the configuration is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags),
you can override hyperparameters.
For example, to change the number of training
steps, and the batch size:

```shell
python -m symphony --config configs/qm9/e3schnet_and_nequip.py \
    --workdir ./workdirs \
    --config.num_train_steps=10 --config.max_n_graphs=16
```

For more extensive changes, directly edit the configuration files,
or add your own.
