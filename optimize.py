import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from surrogate_model import UNet
from topology_loss_functions import circular_density_filter_torch
from plate_models import PlateModel

device = 'cpu'
MINF = 32
FILLW = 1000

def optimize_support_map(model, support_map, target_i, target_j, Lx, Ly, n_steps=1000):
    """Optimize a symmetric SWG support using filtered density and tanh projection."""
    model.eval()
    model.requires_grad_(False)
    losses, losses_ratio_raw, losses_abs_raw = [], [], []

    H, W = support_map.shape[-2], support_map.shape[-1]
    support_map = support_map.detach()
    xs = torch.linspace(0, Lx, W)
    ys = torch.linspace(0, Ly, H)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        dummy_out = model(torch.zeros(1, 3, H, W).to(device)).cpu()
        H_out, W_out = dummy_out.shape[-2], dummy_out.shape[-1]

    mask_waveguide = torch.ones(1, 1, H, W)
    c = H // 2
    mask_waveguide[0, 0, c - 40: c + 40, :] = 0
    mask_traget = torch.zeros(1, 1, H_out, W_out)
    c = W // 2
    mask_traget[0, 0, c - 15: c + 15, c - 25:c + 25] = 1
    mask_not_target = torch.zeros(1, 1, H_out, W_out)
    mask_not_target[0, 0, c - 15: c + 15, :] = 1
    mask_not_target = mask_not_target * (1 - mask_traget)

    existing = torch.clamp(support_map * mask_waveguide, 1e-4, 1 - 1e-4)
    delta = torch.logit(existing).detach().requires_grad_(True)
    optimizer = torch.optim.Adamax([delta], lr=1)
    n_free_pixels = mask_waveguide.sum()

    for i in range(n_steps):
        optimizer.zero_grad()
        supp_soft = torch.sigmoid(delta)
        supp_soft = 0.25 * (supp_soft + supp_soft.flip(-1) + supp_soft.flip(-2) + supp_soft.flip(-2, -1))                     
        min_feature_size_px = MINF
        r_filter_px = max(1.0, min_feature_size_px / 2.0)
        supp_filtered = circular_density_filter_torch(supp_soft, r_filter_px=r_filter_px)
        beta = 2.0 ** (5.0 * i / n_steps)                                                            
        th = np.tanh(beta * 0.5)
        supp_filtered = (th + torch.tanh(beta * (supp_filtered - 0.5))) / (2 * th)
        supp_clip = supp_filtered * mask_waveguide + support_map * (1 - mask_waveguide)
        model_input = torch.cat([supp_clip, grid_x, grid_y], dim=1)
        pred_strain = model(model_input.to(device)).cpu()
        strain_target = pred_strain * mask_traget
        strain_other = pred_strain * mask_not_target
        supp = supp_clip
        H2 = H // 2
        loss_sym_y = F.mse_loss(supp[..., :H2, :], supp[..., H - H2:, :].flip(-2))
        W2 = W // 2
        loss_sym_x = F.mse_loss(supp[..., :W2], supp[..., W - W2:].flip(-1))
        loss_sym = loss_sym_x + loss_sym_y

        eps = 1e-8
        mean_target = strain_target.sum() / (mask_traget.sum() + eps)
        mean_other = strain_other.sum() / (mask_not_target.sum() + eps)
        fill_factor = (supp_clip * mask_waveguide).sum() / n_free_pixels
        fill_loss = (fill_factor - 0.005)**2
        loss_ratio = -(torch.log(mean_target.clamp_min(eps)) - torch.log(mean_other.clamp_min(eps)))
        loss_abs = -mean_target

        if i < 600:
            loss = 1000 * loss_ratio + 10000 * loss_abs
        elif i < 700:
            loss = 1000 * loss_ratio + 10000 * loss_abs + 1000 * loss_sym
        else:
            loss = 1000 * loss_ratio + FILLW * fill_loss + 1000 * loss_sym

        loss.backward()
        optimizer.step()
        losses.append(loss.item()); losses_ratio_raw.append(loss_ratio.item()); losses_abs_raw.append(loss_abs.item())
        if i % 50 == 0:
            print(f'[{i:4d}] ratio={loss_ratio.item():.4f} abs={loss_abs.item():.5f} total={loss.item():.4f} '
                  f'fill={fill_factor.item():.4f}', flush=True)

    with torch.no_grad():
        supp_soft = torch.sigmoid(delta)
        supp_soft = 0.25 * (supp_soft + supp_soft.flip(-1) + supp_soft.flip(-2) + supp_soft.flip(-2, -1))
        supp_filtered = circular_density_filter_torch(supp_soft, r_filter_px=r_filter_px)
        supp_filtered = (th + torch.tanh(beta * (supp_filtered - 0.5))) / (2 * th)
        supp_clip = supp_filtered * mask_waveguide + support_map * (1 - mask_waveguide)
    free_result = (supp_filtered.detach().cpu() > 0.5).float() * mask_waveguide                                   
    result = (free_result + support_map * (1 - mask_waveguide)).squeeze().numpy()
    with torch.no_grad():
        soft = supp_clip.detach().squeeze().numpy()
    return np.clip(result, 0, 1), soft, dict(total=losses, ratio=losses_ratio_raw, abs=losses_abs_raw)



def main():
    """Run SWG optimization and verify the binary result using the FvK solver."""
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--structure', default='data/swg_256.npz')
    p.add_argument('--out', default='outputs/optimized.npz')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--filter-px', type=int, default=32)
    p.add_argument('--q', type=float, default=-6176)
    a = p.parse_args()
    if a.steps < 1 or not 1 <= a.filter_px < 256:
        raise ValueError('Require positive steps and filter-px in [1, 255]')
    global MINF
    MINF = a.filter_px
    torch.manual_seed(42)
    with np.load(a.structure) as data:
        support = data[data.files[0]].squeeze()
    if support.shape != (256, 256):
        raise ValueError('SWG masks are defined for a 256 x 256 grid')
    model = UNet()
    model.load_state_dict(torch.load(a.weights, map_location='cpu', weights_only=True))
    binary, soft, losses = optimize_support_map(model, torch.tensor(support, dtype=torch.float32)[None,None], 128, 128, 12, 12, a.steps)
    result = PlateModel(binary, 12).solve_fvk(a.q, tol=1e-9, maxit=200, strain_tol=3e-3)
    if not result['info']['converged']:
        raise RuntimeError('Final FvK verification did not converge')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, support_map=binary, support_soft=soft,
                        strain_direct_percent=result['strain_memb']*100,
                        loss=losses['total'], q_pa=a.q, width_m=12e-6)

if __name__ == '__main__':
    main()
