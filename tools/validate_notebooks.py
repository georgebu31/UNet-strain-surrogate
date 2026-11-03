import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import nbformat
from nbclient import NotebookClient


def validate(weights=None):
    """Execute standalone notebooks in fresh kernels and keep results under outputs.

    Supply weights to include the surrogate run. Notebook sources remain unchanged;
    output paths and the optional checkpoint path are overridden only in memory.
    """
    root = Path(__file__).resolve().parents[1]
    if weights is not None and not Path(weights).is_file():
        raise FileNotFoundError(f'Checkpoint not found: {weights}')
    output = root / 'outputs/notebook_validation'
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted((root / 'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if path.stem == '05_surrogate' and weights is None:
            reports.append(dict(notebook=path.name, status='skipped', reason='Supply --weights'))
            continue
        print(f'Executing {path.name}', flush=True)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            lines = cell.source.splitlines()
            lines = [('output_file = ' + repr(str(output / (path.stem + '.npz'))))
                     if line.startswith('output_file = ') else line for line in lines]
            if path.stem == '05_surrogate':
                lines = [('weights_file = ' + repr(str(Path(weights).resolve())))
                         if line.startswith('weights_file = ') else line for line in lines]
            cell.source = '\n'.join(lines)
        start = time.time()
        NotebookClient(notebook, timeout=900, startup_timeout=60,
                       resources={'metadata': {'path': str(root / 'notebooks')}}).execute()
        nbformat.write(notebook, output / path.name)
        report = dict(notebook=path.name, status='passed', elapsed_seconds=time.time()-start,
                      code_cells=sum(c.cell_type == 'code' for c in notebook.cells),
                      outputs_redirected=True, checkpoint_override=path.stem == '05_surrogate')
        reports.append(report)
        (output / 'report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        print(report, flush=True)
    (output / 'report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights')
    args = parser.parse_args()
    validate(args.weights)
