# Local validation

Validated on 2026-11-03 with Windows, Python 3.12, NumPy 1.26.4, SciPy 1.12.0 and PyTorch 2.6.0+cpu. The suite limits PyTorch to two CPU threads.

Command: `python -m unittest discover -s tests -v`

Result: **4 tests passed**, 13.286 seconds in the latest rerun.

- FvK zero load, negative deflection under load, convergence and mirror symmetry.
- Exciton diffusion against an exact discrete sine solution, and nonnegative density with strain drift.
- Ten directly generated 32×32 samples, one epoch of the original U-Net, disjoint train/validation/test split, checkpoint reload and finite test error.
- Two topology steps with an analytic toy surrogate, finite losses, exact mirror symmetry, binary output and preservation of the protected region.

`python -m compileall -q article-code` also passed from the parent directory.

## Manuscript regression

`python validate_article.py` completed successfully with **19 checks passed**. Mechanics was recomputed without strain-change early stopping, followed by exciton transport on the fresh strain maps. OMP and MKL thread counts were set to two for this local run.

| Observable | Recomputed result | Reference |
| --- | --- | --- |
| Palacios model strain peak | 1.000022413% | 1.000% (rounded) |
| Branny model strain peak | 0.576429950% | 0.576% (rounded) |
| Kumar model strain peak | 0.118819652% | 0.119% (rounded) |
| Initial SWG FvK strain peak | 0.149497821% | 0.149497821% |
| Optimized SWG FvK strain peak | 0.366745887% | 0.366745887% |
| Initial SWG strain map relative L2 error | 2.96e-15 | 0 |
| Optimized SWG strain map relative L2 error | 3.43e-15 | 0 |
| Original linear strain map max absolute error | 1.01e-13 | 0 |
| Exciton peak gain, 5 K | +37.47472327% | +37.47472327% |
| N_edge(100 nm) gain, 5 K | +8.00979427% | +8.00979427% |
| Exciton peak gain, 300 K | -0.52730798% | -0.52730798% |

The experimental/reference strain values in manuscript Table 1 are 1.0%, 0.6% and 0.3%. Reproducing the model values does not remove its reported experimental discrepancies, particularly the Kumar case. Palacios is a calibration case, not an independent prediction.

`python validate_article.py --transport-only` also passed **8 comparisons**, including absolute initial-SWG peaks of 2.62907226e12 at 5 K and 2.35093384e11 at 300 K under the stored generation normalization. This mode reuses frozen strain maps, so it is a transport regression rather than an independent mechanics check.

`python validate_surrogate.py --weights <manuscript-checkpoint>` passed **4 comparisons** using the existing local `unet_fvk_best.pth` checkpoint:

| Observable | Recomputed result | Manuscript value |
| --- | --- | --- |
| SWG R² | 0.983097780 | 0.983 |
| MAE, percentage points | 0.002820495 | 0.0028 |
| Correlation | 0.992371163 | 0.992 |
| Predicted strain peak | 0.190789789% | 0.19% |

The checkpoint SHA-256 and numerical tolerances are stored in the report. Fixture SHA-256 hashes were verified; compilation passed.

Machine-readable reports are included in [validation/](validation/). Fixture provenance, parameters and tolerance definitions are documented in [data/reference/README.md](data/reference/README.md).

These checks reproduce the supplied manuscript geometries, trained-model inference and transport observables on the 256×256 grid. They do not retrain the full dataset, rerun all 2000 optimization steps, or establish grid-independent peak values. No external datasets or checkpoints were downloaded.

## Standalone notebooks

All five notebooks in `notebooks/` were executed end to end with `nbclient`, each in a fresh Jupyter kernel started from the notebooks directory. Bootstrap imports, editable parameter cells, calculation, NPZ output and rendered plots all passed. The published notebook sources retain clean outputs; executed copies are under the ignored `outputs/notebook_validation/` directory.

| Notebook | Observed peak | Execution time |
| --- | --- | --- |
| `01_linear_fdm.ipynb` | Bending strain 0.230908024260% | 9.85 s |
| `02_linear_plate.ipynb` | Bending strain 0.230908024255% | 4.07 s |
| `03_fvk.ipynb` | Membrane strain 0.149497820680% | 135.16 s |
| `04_excitons.ipynb` | Density 2.629072259802e12 | 7.15 s |
| `05_surrogate.ipynb` | Predicted strain 0.19078979% | 9.80 s |

The nonlinear notebook recomputed SWG mechanics without strain-change early stopping. The exciton notebook independently used the bundled reference strain and 532 nm pump. Surrogate validation supplied the existing local manuscript checkpoint through an in-memory parameter override. Output paths were redirected to the validation directory without changing the notebook sources. Numerical peaks matched their references with relative tolerance 1e-6; all five notebooks produced plot images, and the linear plot was visually inspected.

Reproduce the execution check after installing `requirements-notebooks.txt`:

```bash
python tools/validate_notebooks.py --weights path/to/unet_fvk_best.pth
```

Without `--weights`, the harness runs the four physical notebooks and records the surrogate notebook as skipped. The execution and numerical comparison report is [validation/notebook_validation.json](validation/notebook_validation.json).
