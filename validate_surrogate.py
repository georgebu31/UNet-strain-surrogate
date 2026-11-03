import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from surrogate_model import UNet

ROOT = Path(__file__).resolve().parent


def validate(weights, out):
    """Compare a manuscript checkpoint against the frozen out-of-distribution SWG map."""
    torch.set_num_threads(2)
    with np.load(ROOT/'data/swg_256.npz') as d:
        support = d['array']
    with np.load(ROOT/'data/reference/swg_fvk_reference.npz') as d:
        target = d['strain_A'] * 100
    c = torch.linspace(0, 12, 256)
    gy, gx = torch.meshgrid(c, c, indexing='ij')
    model = UNet().eval()
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    with torch.no_grad():
        pred = model(torch.stack([torch.tensor(support, dtype=torch.float32), gx, gy])[None])[0,0].numpy()
    mse = np.mean((pred-target)**2)
    actual = dict(r2=1-np.sum((pred-target)**2)/np.sum((target-target.mean())**2),
                  mae_percent=np.mean(np.abs(pred-target)),
                  correlation=np.corrcoef(pred.ravel(), target.ravel())[0,1],
                  peak_percent=pred.max())
    reference = dict(r2=(.983,.002), mae_percent=(.0028,.0002),
                     correlation=(.992,.002), peak_percent=(.19,.01))
    checks = []
    for name,(expected,atol) in reference.items():
        checks.append(dict(name=name, actual=float(actual[name]), expected=expected,
                           atol=atol, passed=bool(abs(actual[name]-expected)<=atol)))
        print(checks[-1], flush=True)
    report = dict(checkpoint_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                  checks=checks, passed=all(c['passed'] for c in checks))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report['passed']


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--out', default='outputs/surrogate_validation.json')
    a = p.parse_args()
    raise SystemExit(0 if validate(a.weights, a.out) else 1)
