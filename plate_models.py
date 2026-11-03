"""Linear plate and nonlinear Foppl-von Karman membrane models.

Internal units: length in micrometres, force per length in N/m,
and pressure in MPa. Public solve methods accept pressure in Pa.
Negative w means downward deflection; supports lie at w = 0,
and the optional floor lies at w = -H."""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

E_PA = 167.3e9
NU = 0.19
H_FILM_M = 1e-9

STENCIL = {
    (0, 0): 20,
    (1, 0): -8, (-1, 0): -8, (0, 1): -8, (0, -1): -8,
    (1, 1): 2, (1, -1): 2, (-1, 1): 2, (-1, -1): 2,
    (2, 0): 1, (-2, 0): 1, (0, 2): 1, (0, -2): 1,
}


try:                                                                                    
    from pypardiso import spsolve as _pardiso_solve
except Exception:
    _pardiso_solve = None


def _solve_sym(A, b):
    if _pardiso_solve is not None:
        return _pardiso_solve(A.tocsr(), b)
                                                                                                
    lu = spla.splu(A.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                   options=dict(SymmetricMode=True))
    return lu.solve(b)


class PlateModel:
    """Membrane on a square-grid support map (1: support, 0: suspended).

width_um is the x-domain width; spacing is width_um/(Nx-1).
uv_fixed='supports' fixes in-plane motion on supports and boundaries;
'boundary' allows sliding on supports. D_scale multiplies bending rigidity.
adhesion=None fixes w on supports. Otherwise adhesion=(gamma, h0)
uses U(g)=gamma*(exp(2s)-2*exp(s)), s=sqrt(5)*(1-g^2/h0^2),
g=h0+w, gamma in J/m^2 and h0 in micrometres.
floor_depth optionally extends adhesion to the floor at -H micrometres."""

    def __init__(self, support_map, width_um, E=E_PA, nu=NU, h=H_FILM_M,
                 uv_fixed="supports", D_scale=1.0, adhesion=None, floor_depth=None):
        S = np.where(np.asarray(support_map) > 0.5, 1, 0)
        self.S = S
        self.Ny, self.Nx = S.shape
        self.d = width_um / (self.Nx - 1)
        self.nu = nu
        self.h_um = h * 1e6
        self.Eh = E * h                                                   
        self.D = D_scale * E * h**3 / (12 * (1 - nu**2)) * 1e12                  
        self.C = self.Eh / (1 - nu**2)
        self.G = self.Eh / (2 * (1 + nu))

        interior = np.zeros_like(S, dtype=bool)
        interior[1:-1, 1:-1] = True
        self.adhesion = adhesion
        self.free_w = interior & (S == 0) if adhesion is None else interior.copy()
        self.free_uv = interior & (S == 0) if uv_fixed == "supports" else interior
        self.adh_mask = S[self.free_w] == 1                                                   
        self.z_surf = np.zeros(int(self.free_w.sum()))                                                  
        if adhesion is not None and floor_depth is not None:
            self.z_surf = np.where(S[self.free_w] == 1, 0.0, -floor_depth)
            self.adh_mask = np.ones(int(self.free_w.sum()), dtype=bool)
        self.nw = int(self.free_w.sum())
        self.nuv = int(self.free_uv.sum())

        self.Kb = self._stencil_matrix(STENCIL, self.D / self.d**4)
        lap = {(0, 0): 4, (1, 0): -1, (-1, 0): -1, (0, 1): -1, (0, -1): -1}
        self.Klap = self._stencil_matrix(lap, 1.0 / self.d**2)                                   
        self._build_triangles()

                                              
    def _stencil_matrix(self, stencil, scale):
        Ny, Nx = self.Ny, self.Nx
        idx = -np.ones((Ny, Nx), dtype=np.int64)
        idx[self.free_w] = np.arange(self.nw)
        ii, jj = np.nonzero(self.free_w)
        rows, cols, vals = [], [], []
        for (di, dj), c in stencil.items():
            ni, nj = ii + di, jj + dj
            ok = (ni >= 0) & (ni < Ny) & (nj >= 0) & (nj < Nx)
            k = idx[ni[ok], nj[ok]]
            good = k >= 0
            rows.append(idx[ii[ok], jj[ok]][good])
            cols.append(k[good])
            vals.append(np.full(int(good.sum()), c * scale))
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(self.nw, self.nw))

    def _build_triangles(self):
        """Build four right triangles per node, one per quadrant, each weighted by d^2/4."""
        Ny, Nx, d = self.Ny, self.Nx, self.d
        N = Ny * Nx
        node = np.arange(N).reshape(Ny, Nx)
        I, J = np.meshgrid(np.arange(Ny), np.arange(Nx), indexing="ij")
        p_all, px_all, py_all, sx_all, sy_all = [], [], [], [], []
        for sx in (1, -1):
            for sy in (1, -1):
                ok = (J + sx >= 0) & (J + sx < Nx) & (I + sy >= 0) & (I + sy < Ny)
                p_all.append(node[I[ok], J[ok]])
                px_all.append(node[I[ok], J[ok] + sx])
                py_all.append(node[I[ok] + sy, J[ok]])
                sx_all.append(np.full(int(ok.sum()), sx))
                sy_all.append(np.full(int(ok.sum()), sy))
        p = np.concatenate(p_all); px = np.concatenate(px_all); py = np.concatenate(py_all)
        sx = np.concatenate(sx_all); sy = np.concatenate(sy_all)
        nt = p.size
        self.nt = nt
        self.tri_p = p
        t = np.arange(nt)
        Dx = sp.csr_matrix((np.concatenate([-sx / d, sx / d]), (np.concatenate([t, t]), np.concatenate([p, px]))), shape=(nt, N))
        Dy = sp.csr_matrix((np.concatenate([-sy / d, sy / d]), (np.concatenate([t, t]), np.concatenate([p, py]))), shape=(nt, N))
        fw = np.flatnonzero(self.free_w.ravel())
        fuv = np.flatnonzero(self.free_uv.ravel())
        self.Dxw, self.Dyw = Dx[:, fw].tocsr(), Dy[:, fw].tocsr()
        self.Dxu, self.Dyu = Dx[:, fuv].tocsr(), Dy[:, fuv].tocsr()
        Zu = sp.csr_matrix((nt, self.nuv)); Zw = sp.csr_matrix((nt, self.nw))
                                                            
        self.Gop = sp.bmat([[self.Dxu, Zu, Zw], [self.Dyu, Zu, Zw],
                            [Zu, self.Dxu, Zw], [Zu, self.Dyu, Zw],
                            [Zu, Zu, self.Dxw], [Zu, Zu, self.Dyw]], format="csr")
        self.GopT = self.Gop.T.tocsr()
        self.At = d * d / 4.0
        cnt = np.bincount(p, minlength=N).astype(float)
        self._tri_cnt = cnt

    def _to_grid(self, vec, mask):
        out = np.zeros((self.Ny, self.Nx))
        out[mask] = vec
        return out

    def _tri_to_nodes(self, val):
        return (np.bincount(self.tri_p, weights=val, minlength=self.Ny * self.Nx) / self._tri_cnt).reshape(self.Ny, self.Nx)

                                                                  
    def solve_linear(self, q_pa, T=0.0, H_floor=None, kc=None, maxit=60, verbose=False):
        """Solve D*biharmonic(w)-T*laplacian(w)=q with optional penalty contact w >= -H."""
        q = q_pa * 1e-6
        K = self.Kb + T * self.Klap
        rhs = np.full(self.nw, q)
        if H_floor is None:
            w = _solve_sym(K, rhs)
            return self._pack(w, None, None, info={"iters": 1, "converged": True})
        if kc is None:
            kc = abs(q) / (0.002 * H_floor)
        w = np.maximum(_solve_sym(K, rhs), -H_floor)
        active = np.zeros(self.nw, dtype=bool)
        ok = False
        for it in range(maxit):
            new_active = (w < -H_floor)
            if it == 0:
                new_active = (w <= -H_floor)
            if it > 0 and np.array_equal(new_active, active):
                ok = True
                break
            active = new_active
            A = K + sp.diags(kc * active.astype(float))
            w = _solve_sym(A, rhs - kc * H_floor * active)
            if verbose:
                print(f"  active-set it {it}: contact fraction {active.mean():.3f}, w_min {w.min():.4f}")
        return self._pack(w, None, None, info={"iters": it, "converged": ok, "contact_frac": float(active.mean())})

                                             
    def _membrane(self, x, e0, need_hess=True):
        nuv, C, G, nu, At = self.nuv, self.C, self.G, self.nu, self.At
        g = self.Gop @ x
        nt = self.nt
        ux, uy, vx, vy, wx, wy = (g[k * nt:(k + 1) * nt] for k in range(6))
        exx = ux + 0.5 * wx**2 + e0
        eyy = vy + 0.5 * wy**2 + e0
        gam = uy + vx + wx * wy
        Nxx = C * (exx + nu * eyy)
        Nyy = C * (eyy + nu * exx)
        Nxy = G * gam
        energy = At * 0.5 * np.sum(Nxx * exx + Nyy * eyy + Nxy * gam)
        dg = np.concatenate([Nxx, Nxy, Nxy, Nyy, Nxx * wx + Nxy * wy, Nxy * wx + Nyy * wy]) * At
        grad = self.GopT @ dg
        H = None
        if need_hess:
            one = np.ones(nt); zero = None
                                                                            
            B = {0: {0: one, 4: wx}, 1: {3: one, 5: wy}, 2: {1: one, 2: one, 4: wy, 5: wx}}
            Cm = {(0, 0): C, (1, 1): C, (0, 1): C * nu, (1, 0): C * nu, (2, 2): G}
            blocks = {}
            for (a, b), c in Cm.items():
                for k, bk in B[a].items():
                    for l, bl in B[b].items():
                        blocks[(k, l)] = blocks.get((k, l), 0) + c * bk * bl
            blocks[(4, 4)] = blocks[(4, 4)] + Nxx
            blocks[(5, 5)] = blocks[(5, 5)] + Nyy
            blocks[(4, 5)] = blocks[(4, 5)] + Nxy
            blocks[(5, 4)] = blocks[(5, 4)] + Nxy
            t = np.arange(nt)
            rows = np.concatenate([k * nt + t for (k, l) in blocks])
            cols = np.concatenate([l * nt + t for (k, l) in blocks])
            vals = np.concatenate([At * v for v in blocks.values()])
            W = sp.csr_matrix((vals, (rows, cols)), shape=(6 * nt, 6 * nt))
            H = (self.GopT @ W @ self.Gop).tocsr()
        return energy, grad, H, (exx, eyy, gam, Nxx, Nyy, Nxy)

    def adhesion_terms(self, w):
        """Return adhesion energy density and its first two derivatives in w.

Use U(g)=gamma*(exp(2s)-2*exp(s)), s=sqrt(5)*(1-g^2/h0^2), g=h0+w.
A repulsive wall at g < 0.5*h0 prevents penetration into the support."""
        gamma, h0 = self.adhesion
        r5 = np.sqrt(5.0)
        g = h0 + w
        s = r5 * (1 - g**2 / h0**2)
        ds = -2 * r5 * g / h0**2
        dds = -2 * r5 / h0**2
        e1, e2 = np.exp(s), np.exp(2 * s)
        U = gamma * (e2 - 2 * e1)
        dU = gamma * (2 * e2 - 2 * e1) * ds
        ddU = gamma * ((4 * e2 - 2 * e1) * ds**2 + (2 * e2 - 2 * e1) * dds)
        kw = 100 * 40 * gamma / h0**2
        pen = np.maximum(0.5 * h0 - g, 0.0)
        U = U + 0.5 * kw * pen**2
        dU = dU - kw * pen
        ddU = ddU + kw * (pen > 0)
        return U, dU, ddU

    def _total(self, x, q, e0, H_floor, kc, need_hess=True, stick=None):
        nuv, d2 = self.nuv, self.d**2
        w = x[2 * nuv:]
        Em, gm, Hm, _ = self._membrane(x, e0, need_hess)
        Kw = self.Kb @ w
        energy = Em + d2 * (0.5 * w @ Kw - q * w.sum())
        grad = gm.copy()
        grad[2 * nuv:] += d2 * (Kw - q)
        diag_c = None
        if H_floor is not None:
            pen = np.maximum(-H_floor - w, 0.0)
            energy += d2 * 0.5 * kc * np.sum(pen**2)
            grad[2 * nuv:] -= d2 * kc * pen
            diag_c = d2 * kc * (pen > 0)
        if self.adhesion is not None:                                    
            ea, ga, ha = self.adhesion_terms((w - self.z_surf)[self.adh_mask])
            energy += d2 * ea.sum()
            grad[2 * nuv:][self.adh_mask] += d2 * ga
            diag_a = np.zeros(self.nw)
            diag_a[self.adh_mask] = d2 * ha
            diag_c = diag_a if diag_c is None else diag_c + diag_a
        if stick is not None:                                                                            
            uv = x[:2 * nuv]
            s2 = np.concatenate([stick, stick]) * d2
            energy += 0.5 * np.sum(s2 * uv**2)
            grad[:2 * nuv] += s2 * uv
        H = None
        if need_hess:
            n = x.size
            Kb_big = sp.bmat([[sp.csr_matrix((2 * nuv, 2 * nuv)), None], [None, d2 * self.Kb]], format="csr")
            H = Hm + Kb_big
            if diag_c is not None:
                H = H + sp.diags(np.concatenate([np.zeros(2 * nuv), diag_c]))
            if stick is not None:
                H = H + sp.diags(np.concatenate([s2, np.zeros(self.nw)]))
        return energy, grad, H

    def solve_fvk(self, q_pa, e0=0.0, H_floor=None, kc=None, x0=None, tol=1e-7, maxit=200, verbose=False, stick=None,
                  callback=None, strain_tol=None):
        """Minimize Foppl-von Karman energy with Newton steps and line search.

e0 is isotropic prestrain, giving tension Eh*e0/(1-nu).
H_floor is floor depth in micrometres; kc is the contact penalty stiffness.
strain_tol optionally stops when the relative L2 change of triangle
membrane trace strain falls below this threshold between iterations.
stick assigns in-plane spring stiffness in (N/m)/um^2 at free u,v nodes
to represent sticking in an established contact region."""
        t0 = time.time()
        q = q_pa * 1e-6
        nuv, nw = self.nuv, self.nw
        if H_floor is not None and kc is None:
            kc = abs(q) / (0.002 * H_floor)
        f = lambda x, hess=True: self._total(x, q, e0, H_floor, kc, hess, stick)

        if x0 is None:
            T0 = self.Eh * e0 / (1 - self.nu)
            Kl = (self.Kb + max(T0, 0.0) * self.Klap).tocsr()
            if self.adhesion is None:
                wl = _solve_sym(Kl, np.full(nw, q))
            else:                                                                                             
                fr = ~self.adh_mask
                wl = np.zeros(nw)
                if fr.any():
                    wl[fr] = _solve_sym(Kl[fr][:, fr], np.full(int(fr.sum()), q))
            if H_floor is not None:
                wl = np.maximum(wl, -H_floor)
            best = (np.inf, 0.0)
            for s in np.geomspace(1e-6, 1.0, 60):
                x = np.concatenate([np.zeros(2 * nuv), s * wl])
                En = f(x, False)[0]
                if En < best[0]:
                    best = (En, s)
            x = np.concatenate([np.zeros(2 * nuv), best[1] * wl])
        else:
            x = x0.copy()

        gscale = max(abs(q), 1e-3) * self.d**2                                                   
        En, g, H = f(x)
        hist = []
        converged = False
        for it in range(maxit):
            gn = np.abs(g).max() / gscale
            hist.append((En, gn))
            if callback is not None:                                                                                 
                callback(it, x, time.time() - t0, gn)
            if strain_tol is not None:
                gg = self.Gop @ x
                ntr = self.nt
                tr = gg[:ntr] + gg[3 * ntr:4 * ntr] + 0.5 * (gg[4 * ntr:5 * ntr]**2 + gg[5 * ntr:]**2)
                if it > 0 and np.linalg.norm(tr - tr_prev) < strain_tol * np.linalg.norm(tr):
                    converged = True
                    break
                tr_prev = tr
            if verbose:
                print(f"  it {it:3d}  E = {En:+.6e}  |g|/qd² = {gn:.2e}  w_min = {x[2*nuv:].min():+.4f} um")
            if gn < tol:
                converged = True
                break
            lam = 0.0
            dscale = H.diagonal().mean()
            for attempt in range(12):
                A = H if lam == 0 else H + lam * dscale * sp.identity(x.size, format="csr")
                try:
                    dx = _solve_sym(A, -g)
                    slope = g @ dx
                    if np.all(np.isfinite(dx)) and slope < 0:
                        break
                except RuntimeError:
                    pass
                lam = 1e-8 if lam == 0 else lam * 30
            if abs(slope) < 1e-11 * abs(En) and gn < 1e-3:                                                            
                converged = True
                break
            a = 1.0
            accepted = False
            for ls in range(40):
                xn = x + a * dx
                En_new = f(xn, False)[0]
                if En_new <= En + 1e-4 * a * slope + 1e-13 * abs(En):
                    accepted = True
                    break
                a *= 0.5
            if not accepted:
                if verbose:
                    print("  Line search failed to decrease energy; stopping")
                break
            x = xn
            En, g, H = f(x)
        out = self._pack(x[2 * nuv:], x[:nuv], x[nuv:2 * nuv], e0=e0, x=x,
                         info={"iters": it, "converged": converged, "grad": float(np.abs(g).max() / gscale),
                               "time_s": time.time() - t0, "energy": float(En)})
        if H_floor is not None:
            out["info"]["contact_frac"] = float((x[2 * nuv:] < -H_floor).mean())
        return out

                                         
    def _pack(self, w, u, v, e0=0.0, x=None, info=None):
        d, z = self.d, self.h_um / 2
        W = self._to_grid(w, self.free_w)
        lap = np.zeros_like(W)
        lap[1:-1, 1:-1] = ((W[1:-1, 2:] - 2 * W[1:-1, 1:-1] + W[1:-1, :-2]) +
                           (W[2:, 1:-1] - 2 * W[1:-1, 1:-1] + W[:-2, 1:-1])) / d**2
        gy, gx = np.gradient(W, d)
        out = {"w": W, "strain_bend": -z * lap, "slope": np.hypot(gx, gy),
               "strain_geom": 0.5 * (gx**2 + gy**2), "info": info or {}}
        if u is not None:
            _, _, _, (exx, eyy, gam, Nxx, Nyy, Nxy) = self._membrane(x, e0, need_hess=False)
            out["u"] = self._to_grid(u, self.free_uv)
            out["v"] = self._to_grid(v, self.free_uv)
            out["strain_memb"] = self._tri_to_nodes(exx + eyy)
            out["Nxx"] = self._tri_to_nodes(Nxx)
            out["Nyy"] = self._tri_to_nodes(Nyy)
            out["Nxy"] = self._tri_to_nodes(Nxy)
                                                                                                 
            mean = 0.5 * (out["Nxx"] + out["Nyy"])
            rad = np.sqrt(0.25 * (out["Nxx"] - out["Nyy"])**2 + out["Nxy"]**2)
            out["N1"], out["N2"] = mean + rad, mean - rad
            out["x"] = x
        return out


def summary(res, name=""):
    """Summarize deflection, slope and strain extrema (strain in percent)."""
    s = {"name": name, "w_max_nm": float(-res["w"].min() * 1e3), "slope_max": float(res["slope"].max()),
         "bend_min_%": float(res["strain_bend"].min() * 100), "bend_max_%": float(res["strain_bend"].max() * 100)}
    if "strain_memb" in res:
        s["memb_min_%"] = float(res["strain_memb"].min() * 100)
        s["memb_max_%"] = float(res["strain_memb"].max() * 100)
    else:
        s["geom_max_%"] = float(res["strain_geom"].max() * 100)
    s.update(res["info"])
    return s


def corr(a, b, mask=None):
    """Return the Pearson correlation of two maps over an optional mask."""
    if mask is None:
        mask = np.ones_like(a, dtype=bool)
    x, y = a[mask].ravel(), b[mask].ravel()
    x = x - x.mean(); y = y - y.mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y) + 1e-300))
