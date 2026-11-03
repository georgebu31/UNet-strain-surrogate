import argparse
import json
import random
from pathlib import Path

import numpy as np

import geometries
from plate_models import PlateModel


def generate(out, n=300, grid=256, width_um=12.0, q=-6176.0, classes=None, seed=17):
    """Generate class-balanced FvK maps; store strain in percent and lengths in metres."""
    if n < 1 or grid < 16 or width_um <= 0:
        raise ValueError('Require n >= 1, grid >= 16 and width_um > 0')
    geometries.Ny = geometries.Nx = 256
    geometries.cx = geometries.cy = 128
    names = list(geometries.SCENARIOS) if classes is None else classes
    if any(name not in geometries.SCENARIOS for name in names):
        raise ValueError('Unknown geometry class')
    masks, strains, deflections, ids, samples = [], [], [], [], []
    for name in names:
        cls = list(geometries.SCENARIOS).index(name)
        for j in range(n):
            random.seed(seed + 1_000_003 * cls + 7919 * j)
            mask = np.zeros((256, 256), dtype=int)
            geometries.SCENARIOS[name](mask)
            indices = np.rint(np.linspace(0, 255, grid)).astype(int)
            mask = mask[np.ix_(indices, indices)]
            mask = (mask > 0.5).astype(np.uint8)
            result = PlateModel(mask, width_um).solve_fvk(q, tol=1e-9, maxit=200, strain_tol=3e-3)
            if not result['info']['converged']:
                raise RuntimeError(f'FvK did not converge: {name}, sample {j}')
            masks.append(mask)
            strains.append(result['strain_memb'] * 100)
            deflections.append(result['w'] * 1e-6)
            ids.append(cls)
            samples.append(j)
    shape = (len(masks), 1, grid, grid)
    x, y = np.meshgrid(np.linspace(0, width_um * 1e-6, grid), np.linspace(0, width_um * 1e-6, grid))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, support_map=np.asarray(masks, dtype=np.float32).reshape(shape),
                        strain=np.asarray(strains, dtype=np.float32).reshape(shape),
                        deflection_map=np.asarray(deflections, dtype=np.float32).reshape(shape),
                        X_coord=np.broadcast_to(x, shape).astype(np.float32),
                        Y_coord=np.broadcast_to(y, shape).astype(np.float32),
                        class_id=ids, sample_id=samples,
                        metadata=json.dumps(dict(seed=seed, q_pa=q, width_um=width_um, grid=grid, classes=names)))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate the 21-class FvK strain dataset')
    p.add_argument('--out', default='outputs/dataset.npz')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--grid', type=int, default=256)
    p.add_argument('--width-um', type=float, default=12)
    p.add_argument('--q', type=float, default=-6176)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--classes', nargs='+', choices=list(geometries.SCENARIOS))
    a = p.parse_args()
    generate(a.out, a.n, a.grid, a.width_um, a.q, a.classes, a.seed)
