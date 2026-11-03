import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from exciton_solver_fixed import solve_exciton_fixed
from plate_models import PlateModel

ROOT = Path(__file__).resolve().parent
REF = ROOT / 'data/reference'


def edge_distance(support, refinement=16):
    """Measure distance in nm to closed grating cells on the manuscript's fine raster."""
    c = np.linspace(0, 12, 256)
    teeth = (support > 0) & (np.abs(c-6) < 1)[:, None]
    j = np.arange(255 * refinement + 1)
    lo = np.clip(-((refinement-2*j)//(2*refinement)), 0, 255)
    hi = np.clip((2*j+refinement)//(2*refinement), 0, 255)
    raster = np.zeros((len(j), len(j)), dtype=bool)
    for rows in (lo, hi):
        for cols in (lo, hi):
            raster |= teeth[np.ix_(rows, cols)]
    distance = distance_transform_edt(~raster)
    return distance[::refinement, ::refinement] * (12000/255/refinement)


def validate(out, full=True):
    """Recompute manuscript observables and compare against frozen local references.

    Full mode recomputes mechanics; transport-only mode uses saved strain maps
    and therefore does not independently validate the mechanical solver.
    """
    start = time.time()
    expected = json.loads((REF/'expected.json').read_text())
    report = {'mode': 'full' if full else 'transport-only', 'checks': []}

    def progress(iteration, state, elapsed, gradient):
        if iteration % 10 == 0:
            print(f'  Newton iteration {iteration}, elapsed {elapsed:.1f}s, relative gradient {gradient:.3g}', flush=True)

    def check(name, actual, reference, rtol=1e-5, atol=1e-8):
        passed = bool(np.isclose(actual, reference, rtol=rtol, atol=atol))
        report['checks'].append(dict(name=name, actual=float(actual), expected=float(reference),
                                     rtol=rtol, atol=atol, passed=passed))
        print(f'{name}: {actual:.10g}; expected {reference:.10g}; {"PASS" if passed else "FAIL"}', flush=True)

    with np.load(ROOT/'data/swg_256.npz') as d:
        sa = d['array']
    with np.load(REF/'swg_fvk_reference.npz') as d:
        strains = {'A': d['strain_A'], 'B': d['strain_B']}
        supports = {'A': sa, 'B': d['support_B']}
    if full:
        for label, file, width, target in [
            ('Palacios', 'palasios_2017.npz', 20, 1.000),
            ('Branny', 'branny_2017_WSe2.npz', 40, .576),
            ('Kumar', 'kumar.npz', 6, .119),
        ]:
            with np.load(REF/file) as d:
                support = d['array']
            print(f'Solving {label}...', flush=True)
            r = PlateModel(support, width).solve_fvk(-6176.257131095856, tol=1e-8, maxit=200, callback=progress)
            check(label+' convergence', int(r['info']['converged']), 1, rtol=0, atol=0)
            check(label+' strain peak (%)', r['strain_memb'].max()*100, target, rtol=0, atol=.001)
        print('Solving original linear SWG...', flush=True)
        linear = PlateModel(sa, 12).solve_linear(-33.87)
        with np.load(REF/'strain_swg_linear.npz') as d:
            saved = d['arr_0']
        check('Linear SWG map max absolute error', np.abs(linear['strain_bend']-saved).max(), 0, rtol=0, atol=1e-10)
        for label in ('A', 'B'):
            print(f'Solving FvK SWG {label}...', flush=True)
            r = PlateModel(supports[label], 12).solve_fvk(-6176, tol=1e-9, maxit=200, callback=progress)
            check(label+' convergence', int(r['info']['converged']), 1, rtol=0, atol=0)
            check(label+' strain peak (%)', r['strain_memb'].max()*100,
                  strains[label].max()*100, rtol=1e-4)
            error = np.linalg.norm(r['strain_memb']-strains[label])/np.linalg.norm(strains[label])
            check(label+' strain map relative L2 error', error, 0, rtol=0, atol=1e-4)
            strains[label] = r['strain_memb']
    c = np.linspace(0, 12, 256)
    wg = np.broadcast_to((np.abs(c-6)<.5)[:, None], (256,256))
    pump = np.loadtxt(REF/'pump_532nm_w2.txt')**2
    generation = 1e21*pump/pump.max()
    distance = edge_distance(sa)
    for label, temp, diffusion, lifetime, alpha in [
        ('5K', 5, 2.4e-4, 498e-12, -.0315),
        ('300K', 300, .97e-4, 1.02e-9, -.0315),
    ]:
        kw = dict(D=diffusion, tau=lifetime, mu=diffusion/(8.617e-5*temp), alpha=alpha)
        density = {k: solve_exciton_fixed(e, 12e-6, generation, **kw) for k,e in strains.items()}
        baseline = solve_exciton_fixed(np.zeros((256,256)), 12e-6, generation, **kw)
        observables = {}
        for k,n in density.items():
            observables[k+'_peak'] = n[wg].max()
            observables[k+'_edge100'] = n[(supports[k]==0)&(distance<=100)].sum()*(12/255)**2
        observables['A_peak_unstrained_ratio'] = observables['A_peak']/baseline[wg].max()
        observables['peak_gain_percent'] = (observables['B_peak']/observables['A_peak']-1)*100
        observables['edge100_gain_percent'] = (observables['B_edge100']/observables['A_edge100']-1)*100
        for key, reference in expected[label].items():
            check(label+' '+key, observables[key], reference, rtol=1e-4, atol=1e-5)
    report['passed'] = all(r['passed'] for r in report['checks'])
    report['elapsed_seconds'] = time.time()-start
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report['passed']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--transport-only', action='store_true')
    p.add_argument('--out', default='outputs/article_validation.json')
    a = p.parse_args()
    raise SystemExit(0 if validate(a.out, not a.transport_only) else 1)
