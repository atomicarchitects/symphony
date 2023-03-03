# GMACE

### Instructions

#### On MIT SuperCloud

Clone the repository:

```shell
git clone git@github.com:atomicarchitects/spherical-harmonic-net.git
cd spherical-harmonic-net
```

```shell
module load anaconda/2023a
pip install -r supercloud_requirements.txt
```

Check that installation suceeded by running a short test:

```shell
python -m train_test
```

#### Everywhere Else

Clone the repository:

```shell
git clone git@github.com:atomicarchitects/spherical-harmonic-net.git
cd spherical-harmonic-net
```

Create and activate a virtual environment:

```shell
python -m venv .venv && source .venv/bin/activate
```

Install dependencies with:

```shell
pip install --upgrade pip && pip install -r requirements.txt
```

Check that installation suceeded by running a short test:

```shell
python -m train_test
```

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

If you just want to test training without any dataset downloads,
you can also run the end-to-end training test on the dummy dataset:

```shell
python -m train_test
```

For more extensive changes, you can directly edit the configuration files,
and even add your own.
