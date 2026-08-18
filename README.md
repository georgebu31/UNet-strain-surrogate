# UNet-strain-surrogate

Neural-network surrogate for strain-induced exciton properties in semiconductor nanostructures, used to accelerate topology optimization that would otherwise require repeated finite-difference method (FDM) strain solves and exciton eigenvalue calculations.

## Overview

Computing exciton states as a function of nanostructure geometry normally requires:

1. Solving the elastic strain field with a **finite-difference method (FDM)** solver.
2. Feeding the resulting strain field into an **exciton solver** to obtain eigenvalues/eigenstates.

Both steps are computationally expensive, which makes them impractical inside an iterative topology optimization loop. This project trains a **U-Net surrogate model** to map geometry/strain input directly to the exciton-relevant output, replacing the expensive solver calls during optimization.

## Pipeline

The repository is organized as a three-stage pipeline, plus standalone solver demos:

| Notebook / script | Purpose |
|---|---|
| `demo_fdm_solver.ipynb`, `fdm_solver.py` | Standalone demo and implementation of the FDM strain solver |
| `demo_exciton_solver.ipynb`, `exciton_solver.py` | Standalone demo and implementation of the exciton eigenvalue solver |
| `01_generate_dataset.ipynb`, `strain_dataset.py` | Generates the training dataset by running the FDM + exciton solvers over a range of geometries/strain configurations |
| `02_train_surrogate.ipynb`, `surrogate_model.py` | Trains the U-Net surrogate model on the generated dataset |
| `03_topology_optimization.ipynb`, `topology_loss_functions.py` | Runs topology optimization using the trained surrogate in place of the direct solvers |

## Repository structure

```
.
├── fdm_solver.py                  # FDM strain solver
├── exciton_solver.py              # Exciton eigenvalue solver
├── strain_dataset.py              # Dataset generation / loading utilities
├── surrogate_model.py             # U-Net surrogate architecture
├── topology_loss_functions.py     # Loss functions for topology optimization
├── demo_fdm_solver.ipynb
├── demo_exciton_solver.ipynb
├── 01_generate_dataset.ipynb
├── 02_train_surrogate.ipynb
├── 03_topology_optimization.ipynb
└── .gitignore                     # Excludes generated data (*.npz) and model checkpoints (*.pth)
```

## Requirements

- Python 3.11
- NumPy, SciPy
- PyTorch
- Jupyter

Install dependencies (adjust to your actual `requirements.txt`/environment):

```bash
pip install numpy scipy torch jupyter
```

Or simply open and run each notebook interactively in Jupyter.

## Results
Use "swg_256.npz" file to repeat the article results on the strain distibution, topology optimisation, exciton density. Use "gauss_field_swg.txt" to calculate the exciton density. Use "TE_field.txt" to calculate the overlap of the exciton density and TE mode field.

