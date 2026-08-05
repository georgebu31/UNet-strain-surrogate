import torch
import torch.nn.functional as F


def circular_density_filter_torch(x, r_filter_px):
    r = r_filter_px
    half = int(r)
    size = 2 * half + 1

    yx = torch.arange(-half, half + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(yx, yx, indexing='ij')
    d2 = xx**2 + yy**2

    mask = d2 <= r**2
    weights = r - torch.sqrt(d2)
    weights = torch.where(mask, weights, torch.zeros_like(weights))

    kernel = weights / (weights.sum() + 1e-8)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    pad = half
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(x_padded, kernel)


def tv_loss(x):
    diff_h = x[..., 1:, :] - x[..., :-1, :]
    diff_w = x[..., :, 1:] - x[..., :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()


def connectivity_loss(x, min_neighbors=2):
    kernel = torch.ones(1, 1, 3, 3, device=x.device)
    kernel[0, 0, 1, 1] = 0
    neighbor_sum = F.conv2d(x, kernel, padding=1)
    penalty = x * F.relu(min_neighbors - neighbor_sum)
    return penalty.mean()


def binarization_loss(x):
    return (x * (1.0 - x)).mean()
