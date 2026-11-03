"""Stationary conservative exciton drift-diffusion-recombination solver.

Solve div(D grad(n) + mu n grad(U)) - n/tau + G = 0,
with U = alpha * strain_percent in eV and n = 0 on the boundary.
Mobility is in SI units: 169 cm^2/(eV s) = 1.69e-2 m^2/(eV s).
Scharfetter-Gummel edge fluxes conserve particles and preserve the
Boltzmann equilibrium n proportional to exp(-mu U / D)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

MU_SI = 169e-4                                              
D_SI = 0.97e-4              
TAU = 1.02e-9            
ALPHA = -63.2e-3                                                    


def bernoulli(x):
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-8
    xs = np.where(small, 1.0, x)
    return np.where(small, 1.0 - x / 2, xs / np.expm1(xs))


def gaussian_G(N, width, w_beam=1e-6, G0=1e21, x0=None, y0=None):
    c = np.linspace(0, width, N)
    X, Y = np.meshgrid(c, c)
    x0 = width / 2 if x0 is None else x0
    y0 = width / 2 if y0 is None else y0
    return G0 * np.exp(-((X - x0)**2 + (Y - y0)**2) / w_beam**2)


def solve_exciton_fixed(strain, width, G, D=D_SI, tau=TAU, mu=MU_SI, alpha=ALPHA):
    """Solve for density using dimensionless strain, width in metres and generation map G."""
    Ny, Nx = strain.shape
    dx = width / (Nx - 1)
    U = alpha * strain * 100.0
    N = Ny * Nx
    node = np.arange(N).reshape(Ny, Nx)
    interior = np.zeros((Ny, Nx), dtype=bool)
    interior[1:-1, 1:-1] = True

    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    k = D / dx**2
    for a, b in ((node[:, :-1], node[:, 1:]), (node[:-1, :], node[1:, :])):                      
        a, b = a.ravel(), b.ravel()
        d = mu * (U.ravel()[b] - U.ravel()[a]) / D
        Bp, Bm = bernoulli(d), bernoulli(-d)
                                                                                                                
        rows += [a, a, b, b]; cols += [a, b, b, a]
        vals += [-k * Bp, k * Bm, -k * Bm, k * Bp]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    A = A - sp.identity(N, format="csr") / tau
    rhs = -G.ravel().astype(float)

    free = np.flatnonzero(interior.ravel())
    n = np.zeros(N)
    n[free] = spla.spsolve(A[free][:, free].tocsc(), rhs[free])
    return n.reshape(Ny, Nx)
