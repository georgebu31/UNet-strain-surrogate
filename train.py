import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from strain_dataset import StrainDataset
from surrogate_model import UNet


def loss_function(pred, target):
    """Combine the manuscript's MSE and spatial-gradient losses."""
    grad = F.l1_loss(pred[..., 1:, :]-pred[..., :-1, :], target[..., 1:, :]-target[..., :-1, :])
    grad += F.l1_loss(pred[..., :, 1:]-pred[..., :, :-1], target[..., :, 1:]-target[..., :, :-1])
    return 1000 * F.mse_loss(pred, target) + 10 * grad


def train(dataset, out, epochs=40, batch_size=32, seed=42, device='cpu'):
    """Train with a fixed 80/10/10 split and evaluate the best validation checkpoint."""
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = StrainDataset(dataset)
    if len(ds) < 10 or any(s % 32 for s in ds.support.shape[-2:]):
        raise ValueError('Need >=10 samples and spatial dimensions divisible by 32')
    perm = torch.randperm(len(ds)).tolist()
    n, v = int(.8 * len(ds)), max(1, int(.1 * len(ds)))
    splits = dict(train=perm[:n], valid=perm[n:n+v], test=perm[n+v:])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, history = float('inf'), []
    for epoch in range(epochs):
        record = {}
        for phase in ('train', 'valid'):
            model.train(phase == 'train')
            total = 0
            loader = DataLoader(Subset(ds, splits[phase]), batch_size=batch_size, shuffle=phase == 'train')
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    loss = loss_function(model(x), y)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                total += loss.item() * len(x)
            record[phase] = total / len(splits[phase])
        history.append(record)
        if record['valid'] < best:
            best = record['valid']
            torch.save(model.state_dict(), out / 'best.pth')
        print(epoch + 1, record, flush=True)
    model.load_state_dict(torch.load(out / 'best.pth', map_location=device, weights_only=True))
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in DataLoader(Subset(ds, splits['test']), batch_size=batch_size):
            preds.append(model(x.to(device)).cpu())
            targets.append(y)
    pred, target = torch.cat(preds), torch.cat(targets)
    ss = ((target-target.mean())**2).sum().item()
    report = dict(seed=seed, splits=splits, history=history,
                  test_mse=F.mse_loss(pred, target).item(), test_mae=F.l1_loss(pred, target).item(),
                  test_r2=1-((pred-target)**2).sum().item()/ss if ss else None)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('--out', default='outputs/training')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    train(a.dataset, a.out, a.epochs, a.batch_size, a.seed, a.device)
