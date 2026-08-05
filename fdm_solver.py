import numpy as np
import random
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def get_w_strain(support_map, width=6e-6):
    E = 167.3e9
    nu = 0.19
    rho = 8200
    h = 1e-9
    g = 9.81
    height = width

    support_map = np.where(support_map > 0.5, 1, 0)

    Ny, Nx = support_map.shape
    dx = width / (Nx - 1)
    dy = height / (Ny - 1)

    D_flexural = E * h**3 / (12 * (1 - nu**2))
    q_pressure = -33.87

    x_coords = np.linspace(0, width, Nx)
    y_coords = np.linspace(0, height, Ny)
    X, Y = np.meshgrid(x_coords, y_coords)

    unknown_nodes = []
    node_index = {}
    k = 0
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            if support_map[i, j] == 0:
                unknown_nodes.append((i, j))
                node_index[(i, j)] = k
                k += 1

    N_unk = len(unknown_nodes)

    stencil = {
        (0, 0): 20,
        (1, 0): -8, (-1, 0): -8, (0, 1): -8, (0, -1): -8,
        (1, 1): 2, (1, -1): 2, (-1, 1): 2, (-1, -1): 2,
        (2, 0): 1, (-2, 0): 1, (0, 2): 1, (0, -2): 1,
    }

    scale = D_flexural / dx**4
    K_matrix = lil_matrix((N_unk, N_unk), dtype=float)
    F_vector = np.full(N_unk, q_pressure)

    for idx, (i, j) in enumerate(unknown_nodes):
        for (di, dj), coeff in stencil.items():
            ni, nj = i + di, j + dj
            if (ni, nj) in node_index:
                K_matrix[idx, node_index[(ni, nj)]] = coeff * scale

    w_vec = spsolve(K_matrix.tocsr(), F_vector)

    deflection_map = np.zeros((Ny, Nx))
    for idx, (i, j) in enumerate(unknown_nodes):
        deflection_map[i, j] = w_vec[idx]

    d2wdx2 = np.zeros_like(deflection_map)
    d2wdy2 = np.zeros_like(deflection_map)
    d2wdxdy = np.zeros_like(deflection_map)

    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            d2wdx2[i, j] = (deflection_map[i, j+1] - 2*deflection_map[i, j] + deflection_map[i, j-1]) / dx**2
            d2wdy2[i, j] = (deflection_map[i+1, j] - 2*deflection_map[i, j] + deflection_map[i-1, j]) / dy**2
            d2wdxdy[i, j] = (deflection_map[i+1, j+1] - deflection_map[i+1, j-1] -
                             deflection_map[i-1, j+1] + deflection_map[i-1, j-1]) / (4*dx*dy)

    z = h / 2.0
    C1 = -E * z / (1.0 - nu**2)
    C2 = -E * z / (1.0 + nu)

    sigma_xx = np.zeros_like(deflection_map)
    sigma_yy = np.zeros_like(deflection_map)
    sigma_xy = np.zeros_like(deflection_map)
    sigma_vm = np.zeros_like(deflection_map)

    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            sxx = C1 * (d2wdx2[i, j] + nu * d2wdy2[i, j])
            syy = C1 * (d2wdy2[i, j] + nu * d2wdx2[i, j])
            sxy = C2 * d2wdxdy[i, j]
            sigma_xx[i, j] = sxx
            sigma_yy[i, j] = syy
            sigma_xy[i, j] = sxy
            sigma_vm[i, j] = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

    eps_xx = (sigma_xx - nu * sigma_yy) / E
    eps_yy = (sigma_yy - nu * sigma_xx) / E
    strain = eps_xx + eps_yy

    return strain, X, Y


def draw_circle(mask, cx, cy, r):
    Y, X = np.ogrid[:mask.shape[0], :mask.shape[1]]
    mask[(X - cx)**2 + (Y - cy)**2 <= r**2] = 1


def draw_rect(mask, cx, cy, w, h, angle_deg=0):
    Y, X = np.mgrid[:mask.shape[0], :mask.shape[1]]
    a = np.deg2rad(angle_deg)
    Xr = (X - cx)*np.cos(a) + (Y - cy)*np.sin(a)
    Yr = -(X - cx)*np.sin(a) + (Y - cy)*np.cos(a)
    mask[(np.abs(Xr) <= w/2) & (np.abs(Yr) <= h/2)] = 1


def draw_ellipse(mask, cx, cy, rx, ry, angle_deg=0):
    Y, X = np.mgrid[:mask.shape[0], :mask.shape[1]]
    a = np.deg2rad(angle_deg)
    Xr = (X - cx)*np.cos(a) + (Y - cy)*np.sin(a)
    Yr = -(X - cx)*np.sin(a) + (Y - cy)*np.cos(a)
    mask[(Xr/rx)**2 + (Yr/ry)**2 <= 1] = 1


def draw_parabola_region(mask, cx, cy, amplitude, rotation_deg=0, fill="below"):
    Y, X = np.mgrid[:mask.shape[0], :mask.shape[1]]
    a = np.deg2rad(rotation_deg)
    Xr = (X - cx) * np.cos(a) + (Y - cy) * np.sin(a)
    Yr = -(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)

    Xr_norm = Xr / (mask.shape[1] / 2)
    parabola = amplitude * (mask.shape[0] / 2) * Xr_norm**2

    region = (Yr <= parabola) if fill == "below" else (Yr >= parabola)
    mask[region] = 1


def apply_swg_mask(m):
    Ny, Nx = m.shape
    n_bars = random.randint(2, 11)
    fill_factor = random.uniform(0.2, 0.7)
    bar_h = random.randint(10, 60)
    max_y = max(0, Ny - bar_h)
    y_start = random.randint(0, max_y)
    period = Nx // n_bars
    bar_w = int(period * fill_factor)

    m[:] = 0
    for i in range(n_bars):
        x_start = i * period
        m[y_start:y_start + bar_h, x_start:x_start + bar_w] = 1
