from typing import *
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
import pandas as pd
import sys
sys.path.append("..")
from analyses.metrics import *


generated_paths = {
    "200k": "/home/songk/spherical-harmonic-net/analyses/molecules/generated/silica-allegro-200k-train-steps-all-nov25/fait=1.0/pit=1.0/step=best/molecules",
    "500k": "/home/songk/spherical-harmonic-net/analyses/molecules/generated/silica-allegro-500k-train-steps-all-nov25/fait=1.0/pit=1.0/step=best/molecules",
    "1m": "/home/songk/spherical-harmonic-net/analyses/molecules/generated/silica-allegro-1m-train-steps-all-nov25/fait=1.0/pit=1.0/step=best/molecules",
    "nequip": "/home/songk/spherical-harmonic-net/analyses/molecules/generated/silica-nequip-max-35/fait=1.0/pit=1.0/step=best/molecules",
}

all_generated_molecules = {
    model: get_all_molecules(path) for model, path in generated_paths.items()
}

valid_molecules = {
    model: get_all_valid_molecules(molecules) for model, molecules in all_generated_molecules.items()
}

# Make a dataframe for each model for the validity of the molecules.
validity_df = pd.DataFrame(columns=["model", "validity"])

for model, molecules in valid_molecules.items():
    validity_fraction = compute_validity(all_generated_molecules[model], molecules)
    print(f"{model}: {100 * validity_fraction:0.2f}")

    # Compute bootstrap CI for validity
    all_valid_list = jnp.asarray([check_molecule_validity(molecule) for molecule in all_generated_molecules[model]])
    bootstrap_validity_fractions = []
    for rng_seed in range(1000):
        indices = jax.random.choice(jax.random.PRNGKey(rng_seed), len(all_generated_molecules[model]), shape=(len(all_generated_molecules[model]),), replace=True)
        valid = all_valid_list[indices]
        bootstrap_validity_fractions.append(valid.mean())    
    bootstrap_validity_fractions = jnp.array(bootstrap_validity_fractions)
    print(f"{model}: {100 * bootstrap_validity_fractions.mean():0.2f} +/- {100 * bootstrap_validity_fractions.std():0.2f}")
    
    validity_df = pd.concat(
        [
            validity_df,
            pd.DataFrame.from_records(
                {"model": model, "validity": [validity_fraction]}
            ),
        ],
        ignore_index=True,
    )

sns.barplot(data=validity_df, x="model", y="validity")
plt.title("Validity of Generated Molecules")
plt.xlabel("Model")
plt.ylabel("Validity")
plt.show();
print(validity_df)