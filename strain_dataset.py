import random
import torch
import numpy as np
from torch.utils.data import Dataset


class StrainDataset(Dataset):
    def __init__(self, npz_path, augment=False):
        data = np.load(npz_path)
        self.support    = torch.tensor(data['support_map'],    dtype=torch.float32)
        self.strain     = torch.tensor(data['strain'],         dtype=torch.float32)
        self.grid_x     = torch.tensor(data['X_coord'] * 1e6 ,        dtype=torch.float32)
        self.grid_y     = torch.tensor(data['Y_coord'] * 1e6 ,        dtype=torch.float32)
        self.augment    = augment

    def __len__(self):
        return len(self.strain)

    def _augment(self, x, y):
        support = x[0:1]
        if random.random() > 0.5:
            support = torch.flip(support, dims=[2])
            y = torch.flip(y, dims=[2])
        if random.random() > 0.5:
            support = torch.flip(support, dims=[1])
            y = torch.flip(y, dims=[1])
        k = random.randint(0, 3)
        support = torch.rot90(support, k, dims=[1, 2])
        y = torch.rot90(y, k, dims=[1, 2])
        # recompute coordinate grid for the new orientation
        H, W = support.shape[-2], support.shape[-1]
        xs = torch.linspace(x[1].min(), x[1].max(), W)
        ys = torch.linspace(x[2].min(), x[2].max(), H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        x_out = torch.cat([support, grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0)
        return x_out, y

    def __getitem__(self, idx):
        x = torch.cat([self.support[idx], self.grid_x[idx], self.grid_y[idx]], dim=0)  # (3, H, W)
        y = self.strain[idx]  # (1, H, W)
        if self.augment:
            x, y = self._augment(x, y)
        return x, y
