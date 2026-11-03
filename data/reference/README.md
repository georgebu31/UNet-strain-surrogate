# Manuscript regression fixtures

These files provide numerical regression fixtures for the manuscript calculations, rather than independent experimental validation.

| File | Original source |
| --- | --- |
| `palasios_2017.npz` | `lum_dlya_deformacii/palasios_2017.npz` |
| `branny_2017_WSe2.npz` | `lum_dlya_deformacii/branny_2017_WSe2.npz` |
| `kumar.npz` | `lum_dlya_deformacii/kumar.npz` |
| `strain_swg_linear.npz` | Original repository `strain_swg_256_2.npz` |
| `swg_fvk_reference.npz` | `test_26_fvk_maps.npz` (A), `test_28_fvk_Bsym.npy` (B), `test_28_swg_opt_sym.npz` (B geometry) |
| `pump_532nm_w2.txt` | `pump_fdtd/gauss_field_l532_w2.txt` |
| `expected.json` | `results/test_35_transport_alpha.json`; absolute A peaks from `test_35_grid.json` at 256×256 |

Manuscript Table 1 compares experimental/reference strains of 1.0%, 0.6%, 0.3% with model values 1.00%, 0.58%, 0.12%. The first geometry calibrates the load. The other geometries do not match the experimental values exactly; this release checks reproduction of the stated **model values**, preserving those discrepancies.

Literature geometries use widths 20, 40 and 6 µm, respectively, and pressure −6176.257131095856 Pa. The rounded dataset/SWG pressure is −6176 Pa. SWG strain fixtures contain dimensionless membrane trace strain; linear strain is the original bending strain and belongs to a different mechanical model.

The final exciton example uses a 12 µm square domain on 256×256 nodes, the pump amplitude squared and normalized to peak generation 1e21, and zero boundary density. At 5 K: D=2.4e-4 m²/s, tau=498e-12 s, mu=D/(kB*T), alpha=−0.0315 eV/percent, kB=8.617e-5 eV/K. At 300 K: D=0.97e-4 m²/s and tau=1.02e-9 s, with the same Einstein mobility relation and alpha.

Peak means the maximum density within |y−6 µm|<0.5 µm. N_edge(100 nm) sums density on suspended-film nodes within 100 nm of the original grating teeth, multiplied by the cell area in µm². Distances follow the manuscript's closed-cell refinement on a 16-times finer raster; the same reference teeth are used for both geometries. Density values retain the source's generation normalization.

Full mode allows 0.001 percentage points for the rounded literature-geometry model peaks, 1e-10 dimensionless absolute error for the linear strain map, 1e-4 relative error for FvK peaks and maps, and 1e-4 relative plus 1e-5 absolute error for exciton observables. Saved references are never overwritten by validation. `sha256.json` records fixture hashes.

The manuscript describes the 256×256 exciton peak gain as approximately 38%; the underlying stored comparison is 37.4747232713%. Peak gain depends on grid resolution. This check establishes reproduction on the specified grid, not grid independence.
