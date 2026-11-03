# Differentiable strain surrogate and exciton localization

Code accompanying *Differentiable surrogate modeling for strain engineered exciton localization in hybrid TMD photonic nanostructures*.

The pipeline maps a binary substrate support geometry to membrane strain using a direct mechanical solver, trains a U-Net to predict strain, and uses its geometry gradients for topology optimization. Exciton density is computed separately by solving a stationary drift–diffusion–recombination equation. The network predicts strain, not exciton eigenvalues or density.

## Installation

Python 3.11 or 3.12 is recommended. From this repository directory:

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

CPU execution is supported. For GPU training, install the appropriate PyTorch build and pass `--device cuda`. No proprietary software, downloaded weights, notebook execution or neighbouring folders are needed for the toy tests.

## Files

| File | Purpose |
| --- | --- |
| `plate_models.py` | Direct linear plate and nonlinear Föppl–von Kármán solvers, contact and adhesion options |
| `fdm_solver.py` | Original linear FDM solver and geometry drawing utilities |
| `exciton_solver_fixed.py` | Conservative Scharfetter–Gummel exciton transport solver |
| `geometries.py` | Original 21 geometry classes extracted into a normal Python module |
| `generate_dataset.py` | Reproducible class-balanced FvK dataset generation |
| `strain_dataset.py` | Dataset loading and optional augmentation |
| `surrogate_model.py` | Original five-level U-Net with three inputs and one output |
| `train.py` | Training, best validation checkpoint, held-out test metrics |
| `topology_loss_functions.py` | Density filter and regularization utilities |
| `optimize.py` | Symmetric SWG topology optimization and direct FvK verification |
| `solve.py` | Command-line mechanical and exciton solve |
| `data/` | Original SWG geometry and exported pump/TE fields |
| `tests/` | Small physical checks and pipeline integration tests |
| `notebooks/` | Standalone solver runs with editable parameter cells and result plots |

## Quick local verification

```bash
python -m unittest discover -s tests -v
```

The tests check zero-load and loaded mechanics, an exact discrete manufactured diffusion solution, nonnegative exciton density, generation of ten 32×32 FvK samples, one training epoch with checkpoint reload, and two optimization steps using an analytic toy surrogate. The optimization test checks symmetry, binarity and preservation of the protected waveguide region. It does not establish the accuracy of a trained surrogate or reproduce the manuscript's full optimization.

## Reproduce known manuscript values

```bash
python validate_article.py
```

This recomputes the three Table 1 mechanical examples, the original linear SWG map, FvK strain maps for initial and optimized 256×256 SWG geometries, and exciton transport at 5 K and 300 K. Results and tolerances are written to `outputs/article_validation.json`; a failed comparison returns a nonzero exit code. This can take several minutes on a CPU. Full mechanics uses tight convergence settings without the dataset generator's strain-change early stopping.

For the faster transport regression on frozen strain maps:

```bash
python validate_article.py --transport-only --out outputs/transport_validation.json
```

Known targets are 1.000%, 0.576%, 0.119% model strain for Palacios/Branny/Kumar; 0.1494978% and 0.3667459% FvK strain peaks for initial/optimized SWG; and a 37.4747% exciton peak increase with an 8.00979% increase of N_edge(100 nm) at 5 K. These use the final manuscript's 532 nm pump and alpha=31.5 meV/percent, rather than the historical pump and transport defaults. All fixtures, parameter definitions, provenance and tolerances are documented in [data/reference/README.md](data/reference/README.md).

The manuscript's trained U-Net can also be checked against the out-of-distribution SWG example:

```bash
python validate_surrogate.py --weights path/to/unet_fvk_best.pth
```

This checks the reported R²=0.983, correlation=0.992, MAE=0.0028 percentage points and predicted peak strain of 0.19%. The checkpoint is supplied separately and is not embedded in Git. The local validation used the existing manuscript checkpoint; its SHA-256 is recorded in the report. A newly trained or unrelated checkpoint is not expected to reproduce these values. This inference check does not retrain the full manuscript dataset.

## Generate and train

```bash
python generate_dataset.py --out outputs/dataset.npz --n 300 --grid 256
python train.py outputs/dataset.npz --out outputs/training --epochs 40 --batch-size 32
```

Generation defaults: 21 classes, 300 samples per class, 12 µm square domain, pressure −6176 Pa, seed 17. A nonconverged solve raises an error rather than silently entering the dataset. Generation writes the final file at completion; this simple release CLI does not implement distributed merging or restart checkpoints. Full generation and training can require substantial time and RAM. Reduce batch size to fit available memory.

For a small standalone run:

```bash
python generate_dataset.py --out outputs/toy.npz --n 10 --grid 32 --width-um 2 --q -100 --classes single_circle_center
python train.py outputs/toy.npz --epochs 1 --batch-size 2 --out outputs/toy-training
```

Geometry parameters are defined in pixels on the original 256×256 grid. Other grid sizes resample those masks with nearest-neighbour indices and are intended for smoke tests. Training requires dimensions divisible by 32 and at least ten samples. The fixed seed controls geometry sampling and the 80/10/10 sample split. `metrics.json` records split indices, learning curves and test MSE, MAE and R². `best.pth` contains the original U-Net state dict. GPU results need not be bitwise reproducible across devices.

The loss is `1000*MSE + 10*gradient_L1`; Adam uses learning rate 0.001. Validation selects the checkpoint; the held-out test set is evaluated once after training.

## Use each solver separately

Open the standalone notebooks in [notebooks/](notebooks/). From the repository root:

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyterlab
```

Select the installed Python environment as the notebook kernel. Edit the **Parameters** cell, then use **Restart Kernel and Run All Cells**. Each notebook loads the supplied SWG geometry, saves its result under `outputs/` and displays the calculated map.

| Notebook | Run | What to change |
| --- | --- | --- |
| [01_linear_fdm.ipynb](notebooks/01_linear_fdm.ipynb) | Original linear FDM | Input map, domain size, output path |
| [02_linear_plate.ipynb](notebooks/02_linear_plate.ipynb) | Configurable linear plate | Input map, domain size, material, pressure and tension |
| [03_fvk.ipynb](notebooks/03_fvk.ipynb) | Nonlinear FvK mechanics | Input map, domain size, material, pressure, prestrain and boundary condition |
| [04_excitons.ipynb](notebooks/04_excitons.ipynb) | Exciton transport | Strain file, optical pump and transport parameters; defaults to the bundled SWG strain map |
| [05_surrogate.ipynb](notebooks/05_surrogate.ipynb) | U-Net strain prediction | Input map, domain size and matching trained checkpoint (required separately) |

For your own geometry, replace the NPZ input path/key and set its physical size. `width_um` sets both extents of the square domain; modify the support pixels to change individual feature dimensions. The exciton notebook can also read the result saved by the FvK notebook. Units and expected SWG results are described inside each notebook.

## Combined command-line solve and optimization

`solve.py` runs FvK mechanics followed by exciton transport with the historical transport defaults and a synthetic Gaussian pump. It requires a square support array in an NPZ file; `--key` selects its array, or the first array is used. No weights are needed:

```bash
python solve.py --structure data/swg_256.npz --key array --width-um 12 --q -6176 --out outputs/direct.npz
```

The saved NPZ contains `support_map`, dimensionless `strain_fraction`, `deflection_m` in metres, `generation`, `density`, `width_m` and `q_pa`. To use your own geometry, replace the structure path, key and width; a support map from a dataset must first be reduced to one 2D sample. The combined command does not select the final manuscript's 5 K parameters; use [04_excitons.ipynb](notebooks/04_excitons.ipynb) or the article validation command for that case.

Optimization additionally requires trained weights:

```bash
python optimize.py --weights outputs/training/best.pth --steps 2000 --out outputs/optimized.npz
```

Optimization uses the 256×256 SWG masks, a 12 µm domain, Adamax, mirror symmetry, a density filter, progressive tanh projection and the staged objective from the revised local SWG experiment. The middle 80 rows are protected. Thresholding uses the filtered projected density; the final binary result is checked with the direct FvK solver. The log-ratio objective clamps nonpositive predictions to avoid invalid logarithms; sustained clamping indicates an unsuitable surrogate. Use weights trained on the same mechanics, units and domain. A toy checkpoint only verifies execution.

The mechanics solver uses µm internally and accepts pressure in Pa. `strain_memb` is dimensionless. Dataset arrays `support_map`, `strain`, `deflection_map`, `X_coord`, `Y_coord` have shape `(N,1,H,W)`: strain is in percent, lengths are in metres. `StrainDataset` converts coordinate channels to µm. U-Net inputs are `[support, x_um, y_um]`; its output is strain in percent. The exciton solver accepts dimensionless strain and domain width in metres, with zero density on the boundary. Default transport parameters are defined in `exciton_solver_fixed.py`.

`solve.py` uses a synthetic Gaussian pump. The exported `gauss_field_swg.txt` and `TE_field.txt` are provided as reference inputs; the quick CLI does not parse these external field exports or reproduce their overlap integral. Full experimental reproduction requires the matching optical field normalization and parameter choices.

## Provenance and external artifacts

The default mechanical model is nonlinear FvK; the linear FDM solver is also provided. The exciton transport model uses SI mobility and conservative fluxes. U-Net parameter names are retained for checkpoint compatibility. Results from the linear and nonlinear mechanical models should not be treated as interchangeable.

The original README refers to [dataset DOI 10.5281/zenodo.21805899](https://doi.org/10.5281/zenodo.21805899) and [weights DOI 10.5281/zenodo.22077178](https://doi.org/10.5281/zenodo.22077178). These archives are not bundled or downloaded here. Verify their mechanical model and units before using them with this release.

Generated arrays and checkpoints belong under `outputs/` and are ignored by Git. No license has been inferred or assigned: the authors should select the appropriate code and data licenses before public distribution.
