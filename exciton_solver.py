import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def solve_exciton_diffusion(strain, width, height, x_beam=6e-6, y_beam=6e-6, w_beam=1e-6,
                             G_field=None, D=0.97e-4, tau=1e-9, mu=167*1e2, alpha=-63.2e-3,
                             G0_peak=1e21):
    """
    strain: 2D strain map (Ny, Nx)
    width, height: physical domain size, m
    x_beam, y_beam, w_beam: Gaussian excitation beam center and width, m
        (used only if G_field is None)
    G_field: precomputed generation-rate map, e.g. from an FDTD-computed field
        intensity; overrides the Gaussian beam if provided
    Returns the steady-state exciton density map n(x, y).
    """
    Ny, Nx = strain.shape

    # D     = 0.97e-4    # m^2/s
    # tau   = 1e-9    # s
    # kT = 8.617e-5 * 4  # = 0.000345 eV = 0.345 meV     # eV
    # mu    = D / kT     # m^2/(eV*s)
    # alpha = -63.2e-3      # eV/%
    # D     = 2.9e-4    # m^2/s
    # tau   = 1e-12   # s
    # kT = 8.617e-5 * 5  # = 0.000345 eV = 0.345 meV    # eV
    # mu    = D / kT     # m^2/(eV*s)
    # alpha = 63.2e-3      # eV/%
    # D     = 14.5e-4    # m^2/s
    # tau   = 9e-11   # s
    # kT = 8.617e-5 * 300  # = 0.000345 eV = 0.345 meV    # eV
    # mu    = D / kT     # m^2/(eV*s)

    phi = alpha * strain * 100
    dx = width  / (Nx - 1)
    dy = height / (Ny - 1)
    x_coords = np.linspace(0, width,  Nx)
    y_coords = np.linspace(0, height, Ny)
    X, Y = np.meshgrid(x_coords, y_coords)
    d = dx

    dphi_dx = np.zeros_like(phi)
    dphi_dy = np.zeros_like(phi)
    dphi_dx[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2*dx)
    dphi_dy[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2*dx)
    d2phi_dx2 = np.zeros_like(phi)
    d2phi_dy2 = np.zeros_like(phi)
    lap_phi = d2phi_dx2 + d2phi_dy2
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            d2phi_dx2[i, j]  = (phi[i, j+1] - 2*phi[i, j] + phi[i, j-1]) / dx**2
            d2phi_dy2[i, j]  = (phi[i+1, j] - 2*phi[i, j] + phi[i-1, j]) / dy**2

    # G0 = 1e19
    # G  = np.full((Nx, Ny), G0)
    if G_field is not None:
        G = G_field
    else:
        G = G0_peak * np.exp(-((X - x_beam)**2 + (Y - y_beam)**2) / w_beam**2)

    N = Nx * Ny
    A = lil_matrix((N, N))
    b = np.zeros(N)

    def idx(i, j): return i * Ny + j

    for i in range(Nx):
        for j in range(Ny):
            k = idx(i, j)
            if i == 0 or i == Nx-1 or j == 0 or j == Ny-1:
                A[k, k] = 1.0
                b[k]    = 0.0
                continue

            # Fx = mu * dphi_dx[i, j]
            # Fy = mu * dphi_dy[i, j]
            Fx = -mu * dphi_dx[i, j]
            Fy = -mu * dphi_dy[i, j]

            # A[k, idx(i+1, j)] += D/dx**2 - max(-Fx, 0)/dx
            # A[k, idx(i-1, j)] += D/dx**2 - max( Fx, 0)/dx
            A[k, k] -= 2*D/dx**2 + abs(Fx)/dx
            A[k, idx(i+1,j)] += D/dx**2 + max(-Fx, 0)/dx   # was: -
            A[k, idx(i-1,j)] += D/dx**2 + max( Fx, 0)/dx   # was: -

            # A[k, idx(i, j+1)] += D/dx**2 - max(-Fy, 0)/dx
            # A[k, idx(i, j-1)] += D/dx**2 - max( Fy, 0)/dx
            A[k, idx(i, j+1)] += D/dx**2 + max(-Fy, 0)/dx
            A[k, idx(i, j-1)] += D/dx**2 + max( Fy, 0)/dx
            A[k, k] -= 2*D/dx**2 + abs(Fy)/dx
            A[k, k] -= mu * lap_phi[i, j]
            A[k, k] -= 1.0/tau

            b[k] = -G[i, j]

    n_vec = spsolve(A.tocsr(), b)
    n = n_vec.reshape(Nx, Ny)
    return n
