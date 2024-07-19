## The Symphony Tutorial

Interested in learning about Symphony?
The notebooks in this repository will guide you through the building blocks: e3nn-jax, spherical harmonics and $E(3)$-equivariant neural networks.

Slides for our tutorial can be found [here](https://docs.google.com/presentation/d/1a74RRHP_EZfErixEn8T3thUVTvEEg9aRp6Zvz-5UtWM/edit?usp=sharing).


## Local Setup Installation for Notebook 04

- Clone repository.
```bash
!git clone https://github.com/atomicarchitects/symphony.git --depth 1 --branch tutorial
```

Go to the repository:
```bash
cd symphony
```

- Install `symphony`:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
```

Open the notebook:
```bash
jupyter notebook tutorial/04_qm9_playground.ipynb
```