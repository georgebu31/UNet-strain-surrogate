import argparse
from pathlib import Path

import numpy as np

from exciton_solver_fixed import gaussian_G, solve_exciton_fixed
from plate_models import PlateModel


def main():
    """Solve the mechanical and exciton transport problems on a square support map."""
    p = argparse.ArgumentParser()
    p.add_argument('--structure', default='data/swg_256.npz')
    p.add_argument('--key')
    p.add_argument('--width-um', type=float, default=12)
    p.add_argument('--q', type=float, default=-6176)
    p.add_argument('--out', default='outputs/direct.npz')
    a = p.parse_args()
    with np.load(a.structure) as data:
        support = data[a.key or data.files[0]].squeeze()
    if support.ndim != 2 or support.shape[0] != support.shape[1] or a.width_um <= 0:
        raise ValueError('Require a square 2D support map and positive width')
    result = PlateModel(support, a.width_um).solve_fvk(a.q, tol=1e-9, maxit=200, strain_tol=3e-3)
    if not result['info']['converged']:
        raise RuntimeError('FvK solve did not converge')
    width = a.width_um * 1e-6
    generation = gaussian_G(support.shape[0], width)
    density = solve_exciton_fixed(result['strain_memb'], width, generation)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, support_map=support, strain_fraction=result['strain_memb'],
                        deflection_m=result['w']*1e-6, generation=generation, density=density,
                        width_m=width, q_pa=a.q)


if __name__ == '__main__':
    main()
