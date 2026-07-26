"""
2D pehill feature extraction for the OpenFOAM-coupled TBNN.

Builds the same 15-feature input vector used at training time
(see phase2_stress/dataset.py and data/add_pehill_*_features.py).

Inputs are OpenFOAM cell-centred fields (U, k, omega, nu_t) plus the
mesh cell centres (x, y). Outputs the (N, 17) clipped/normalised
feature array and the (N, 10, 9) Frobenius-normalised tensor basis ready
to feed a TBNN trained with `signed_log_lambdas=true` and the 10-extra
feature list.

The wall-distance preprocessing (KDTree + local u_τ) and the four physics
markers are computed identically to data/add_pehill_*_features.py.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


BETA_STAR = 0.09        # ε = β* · k · ω in k–ω SST


# ── 3×3 tensor algebra on (N,3,3) arrays ─────────────────────────────────────
def _mm(A, B):     return np.einsum("nij,njk->nik", A, B)
def _tr(A):        return A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]
def _sym(A):       return 0.5 * (A + A.transpose(0, 2, 1))
def _anti(A):      return 0.5 * (A - A.transpose(0, 2, 1))
def _eye(N):
    I = np.zeros((N, 3, 3))
    I[:, 0, 0] = I[:, 1, 1] = I[:, 2, 2] = 1.0
    return I


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


# ── polyMesh parsing ─────────────────────────────────────────────────────────
def _parse_block_count(text: str) -> tuple[int, int]:
    m = re.search(r"^(\d+)\s*\n\(", text, re.MULTILINE)
    return int(m.group(1)), m.end()


def parse_points(path: Path) -> np.ndarray:
    """Return (Npts, 3) array of mesh vertex coordinates."""
    text = path.read_text()
    N, start = _parse_block_count(text)
    end = text.index("\n)", start)
    rows = re.findall(r"\(([^)]+)\)", text[start:end])
    return np.array([[float(v) for v in r.split()] for r in rows[:N]],
                    dtype=np.float64)


def parse_faces(path: Path) -> list[list[int]]:
    text = path.read_text()
    N, start = _parse_block_count(text)
    end = text.index("\n)", start)
    body = text[start:end]
    matches = re.findall(r"\d+\(([^)]+)\)", body)
    return [[int(x) for x in f.split()] for f in matches[:N]]


def parse_owner(path: Path) -> np.ndarray:
    text = path.read_text()
    N, start = _parse_block_count(text)
    end = text.index("\n)", start)
    return np.array([int(v) for v in text[start:end].split()][:N], dtype=np.int64)


def parse_boundary(path: Path) -> dict:
    """Return {patch_name: {'type': str, 'startFace': int, 'nFaces': int}}."""
    text = path.read_text()
    blocks = re.findall(
        r"(\w+)\s*\{([^}]*)\}", text[text.index("//"):], re.DOTALL
    )
    out = {}
    for name, body in blocks:
        if name == "FoamFile":
            continue
        d = {}
        for kv in re.finditer(r"(\w+)\s+(\S+?);", body):
            d[kv.group(1)] = kv.group(2)
        if "type" in d and "startFace" in d:
            out[name] = {
                "type": d["type"],
                "startFace": int(d["startFace"]),
                "nFaces": int(d["nFaces"]),
            }
    return out


def cell_centres_from_polymesh(case_dir: Path) -> tuple[np.ndarray, dict]:
    """
    Compute cell-centre coordinates by averaging face centres of all faces
    that have a given cell as the owner. Returns (N_cells, 3) and the
    parsed boundary dict.
    """
    pm = case_dir / "constant" / "polyMesh"
    pts = parse_points(pm / "points")
    faces = parse_faces(pm / "faces")
    owners = parse_owner(pm / "owner")
    boundary = parse_boundary(pm / "boundary")

    N_cells = int(owners.max()) + 1
    cell_sum = np.zeros((N_cells, 3))
    cell_cnt = np.zeros(N_cells, dtype=np.int64)
    for fi, owner in enumerate(owners):
        if owner < 0:
            continue
        verts = faces[fi]
        fc = pts[verts].mean(axis=0)
        cell_sum[owner] += fc
        cell_cnt[owner] += 1
    cnt_safe = np.maximum(cell_cnt, 1)[:, None]
    return cell_sum / cnt_safe, boundary


# ── Wall-distance via KDTree (matches data/add_pehill_wall_features.py) ──────
def wall_distance(cell_xy: np.ndarray, n_bins: int = 200) -> np.ndarray:
    """KDTree distance from each cell to nearest bottom-hill or top-wall point."""
    x = cell_xy[:, 0]; y = cell_xy[:, 1]
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    bx, by = [], []
    for i in range(n_bins):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        if sel.sum() < 5:
            continue
        idx = np.argmin(y[sel])
        bx.append(x[sel][idx]); by.append(y[sel][idx])
    bot_x, bot_y = np.asarray(bx), np.asarray(by)
    y_top = y.max()
    top_x = np.linspace(x.min(), x.max(), 200)
    top_y = np.full_like(top_x, y_top)
    wall_pts = np.column_stack(
        [np.concatenate([bot_x, top_x]), np.concatenate([bot_y, top_y])]
    )
    tree = cKDTree(wall_pts)
    d, _ = tree.query(cell_xy, k=1)
    return d, y_top


# ── Velocity-gradient on unstructured 2D grid via least-squares ──────────────
def div_tensor_2D_lsq(xy: np.ndarray, T: np.ndarray, k_nn: int = 12) -> np.ndarray:
    """
    LSQ divergence of a (N,3,3) tensor field on a 2D unstructured mesh.

    Returns (N, 3): div_i = ∂T_i0/∂x + ∂T_i1/∂y  for i = 0,1,2.

    Uses the same nearest-neighbour LSQ stencil as `grad_2D_lsq`, so the
    body force −∇·R produced from a tensor R is consistent with the
    velocity gradient used to build R itself.
    """
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=min(k_nn, len(xy)))
    N = len(xy)
    out = np.zeros((N, 3))
    for c in range(N):
        nbrs = idx[c]
        dx = xy[nbrs] - xy[c]
        if dx.shape[0] < 3:
            continue
        # Solve once per component-pair: G shape (2, 6) where columns are
        # gradients of T[:, 0,0], T[:, 0,1], T[:, 1,0], T[:, 1,1], T[:, 2,0], T[:, 2,1]
        dT_cols = np.stack([
            T[nbrs, 0, 0] - T[c, 0, 0],   # ∂T00/∂(x,y)
            T[nbrs, 0, 1] - T[c, 0, 1],
            T[nbrs, 1, 0] - T[c, 1, 0],
            T[nbrs, 1, 1] - T[c, 1, 1],
            T[nbrs, 2, 0] - T[c, 2, 0],
            T[nbrs, 2, 1] - T[c, 2, 1],
        ], axis=1)
        G, *_ = np.linalg.lstsq(dx, dT_cols, rcond=None)   # (2, 6)
        out[c, 0] = G[0, 0] + G[1, 1]    # ∂T00/∂x + ∂T01/∂y
        out[c, 1] = G[0, 2] + G[1, 3]    # ∂T10/∂x + ∂T11/∂y
        out[c, 2] = G[0, 4] + G[1, 5]    # ∂T20/∂x + ∂T21/∂y
    return out


def grad_2D_lsq(xy: np.ndarray, U2: np.ndarray, k_nn: int = 12) -> np.ndarray:
    """
    Compute (N, 3, 3) velocity gradient using k_nn nearest-neighbour least
    squares. Only x,y components nonzero (z assumed empty).
    xy : (N, 2) cell centres
    U2 : (N, 2) (U, V) at each cell
    """
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=min(k_nn, len(xy)))
    N = len(xy)
    grad = np.zeros((N, 3, 3))
    for i in range(N):
        nbrs = idx[i]
        dx = xy[nbrs] - xy[i]
        dU = U2[nbrs] - U2[i]
        if dx.shape[0] < 3:
            continue
        G, *_ = np.linalg.lstsq(dx, dU, rcond=None)   # (2, 2)
        grad[i, 0, 0] = G[0, 0]   # ∂U/∂x
        grad[i, 0, 1] = G[1, 0]   # ∂U/∂y
        grad[i, 1, 0] = G[0, 1]   # ∂V/∂x
        grad[i, 1, 1] = G[1, 1]   # ∂V/∂y
    return grad


# ── Pope invariants and tensor basis ─────────────────────────────────────────
def _pope_invariants(S_hat, O_hat) -> np.ndarray:
    S2 = _mm(S_hat, S_hat); S3 = _mm(S2, S_hat)
    O2 = _mm(O_hat, O_hat)
    O2S = _mm(O2, S_hat); O2S2 = _mm(O2, S2)
    return np.stack([_tr(S2), _tr(O2), _tr(S3), _tr(O2S), _tr(O2S2)], axis=1)


def _tensor_basis(S_hat, O_hat) -> np.ndarray:
    N = len(S_hat); I = _eye(N)
    S2 = _mm(S_hat, S_hat); O2 = _mm(O_hat, O_hat)
    SO = _mm(S_hat, O_hat); OS = _mm(O_hat, S_hat)
    SO2 = _mm(S_hat, O2);    O2S = _mm(O2, S_hat)
    OS2 = _mm(O_hat, S2);    S2O = _mm(S2, O_hat)
    T1  = S_hat
    T2  = SO - OS
    T3  = S2 - (1/3) * _tr(S2)[:, None, None] * I
    T4  = O2 - (1/3) * _tr(O2)[:, None, None] * I
    T5  = OS2 - S2O
    T6  = O2S + SO2 - (2/3) * _tr(_mm(S_hat, O2))[:, None, None] * I
    T7  = _mm(_mm(O_hat, S_hat), O2) - _mm(_mm(O2, S_hat), O_hat)
    T8  = _mm(_mm(S_hat, O2), S2) - _mm(_mm(S2, O2), S_hat)
    T9  = _mm(O2, S2) + _mm(S2, O2) - (2/3) * _tr(_mm(S2, O2))[:, None, None] * I
    T10 = _mm(_mm(O_hat, S2), O2) - _mm(_mm(O2, S2), O_hat)
    T = np.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], axis=1)  # (N,10,3,3)
    return T.reshape(N, 10, 9)


# ── Main: build full pehill feature pack from OF fields ──────────────────────
def build_pehill_features(
    cell_xy:    np.ndarray,        # (N, 2)
    U:          np.ndarray,        # (N,)
    V:          np.ndarray,        # (N,)
    k:          np.ndarray,        # (N,)
    omega:      np.ndarray,        # (N,)
    nu_t:       np.ndarray,        # (N,)  eddy viscosity from OF
    nu:         float,             # molecular viscosity
    stats:      dict,              # checkpoint stats (mu, sigma, lam_lo, lam_hi,
                                   # signed_log_lambdas, extra_features)
    domain_Lx:  float = 9.0,       # streamwise period (alpha=1.0 default)
    gradU_override: np.ndarray | None = None,  # (N,3,3) — replaces grad_2D_lsq if given
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns (feat_norm (N, n_feat), T_basis (N, 10, 9), aux_dict).

    aux_dict contains intermediate quantities useful for the coupling driver:
      d_wall, u_tau_ref, S_norm, ...
    """
    N = len(U)
    x, y = cell_xy[:, 0], cell_xy[:, 1]
    eps = np.maximum(BETA_STAR * k * omega, 1e-12)

    # 1. Velocity gradient and Pope invariants
    UV2 = np.column_stack([U, V])
    gradU = gradU_override if gradU_override is not None else grad_2D_lsq(cell_xy, UV2)
    S_raw = _sym(gradU); O_raw = _anti(gradU)
    k_over_eps = np.clip(k / eps, 0.0, 100.0)[:, None, None]
    S_hat = k_over_eps * S_raw; O_hat = k_over_eps * O_raw
    lambdas = _pope_invariants(S_hat, O_hat)              # (N, 5)
    T = _tensor_basis(S_hat, O_hat)                       # (N, 10, 9)

    # 2. Wall geometry
    d_wall, y_top = wall_distance(cell_xy)
    half_height = 0.5 * y_top
    # Local u_τ from velocity gradient at near-wall cells
    near = d_wall < 0.01
    if near.sum() < 10:
        # fallback: use mean
        u_tau_ref = np.sqrt(nu * np.abs(gradU[:, 0, 1]).mean())
    else:
        u_tau_ref = float(np.median(np.sqrt(nu * np.abs(gradU[near, 0, 1]))))
    y_plus = d_wall * u_tau_ref / nu
    y_delta = d_wall / max(half_height, 1e-9)

    # 3. Curvature
    speed_sq = U * U + V * V
    speed = np.sqrt(speed_sq)
    speed_floor = np.maximum(speed, 0.05 * speed.mean())
    a_x = U * gradU[:, 0, 0] + V * gradU[:, 0, 1]
    a_y = U * gradU[:, 1, 0] + V * gradU[:, 1, 1]
    kappa = (U * a_y - V * a_x) / (speed_floor ** 3)
    curvature = signed_log1p(kappa)

    # 4. q_balance
    S_xx = gradU[:, 0, 0]; S_yy = gradU[:, 1, 1]
    S_xy = 0.5 * (gradU[:, 0, 1] + gradU[:, 1, 0])
    O_xy = 0.5 * (gradU[:, 1, 0] - gradU[:, 0, 1])
    S2_norm = S_xx ** 2 + S_yy ** 2 + 2.0 * S_xy ** 2     # SᵢⱼSᵢⱼ
    O2_norm = 2.0 * O_xy ** 2                              # |Ω|² (positive)
    q_balance = (O2_norm - S2_norm) / (S2_norm + O2_norm + 1e-30)

    # 5. Streamwise phase  (Lx = 9 for alpha=1.0)
    phase = 2.0 * np.pi * (x - x.min()) / domain_Lx
    cos_x = np.cos(phase); sin_x = np.sin(phase)

    # 6. Physics-of-separation markers
    decel = -a_x / np.maximum(speed_sq, (0.05 * speed.mean()) ** 2)
    decel_marker = signed_log1p(decel)
    tke_to_meanke = k / (k + 0.5 * speed_sq + 1e-30)
    visc_iner = nu / (speed_floor * np.maximum(d_wall, 1e-6))
    viscous_inertia = signed_log1p(visc_iner)
    tau_t = np.where(eps > 1e-12, k / eps, 0.0)
    tsr = tau_t * np.sqrt(2.0 * S2_norm)
    turb_strain_ratio = np.log1p(np.clip(tsr, 0.0, 1e8))

    # 7. Recirculation diagnostics
    abs_U = np.abs(U)
    reverse_flow = U / (abs_U + 0.05 * abs_U.mean() + 1e-12)
    log_nu_t_ratio = np.log1p(np.clip(nu_t / nu, 0.0, 1e8))
    P = nu_t * 2.0 * S2_norm
    pe = np.where(eps > 1e-12, P / eps, 0.0)
    prod_diss_ratio = signed_log1p(np.clip(pe, -1e8, 1e8))

    # 8. Assemble local (non-streamline) extras
    extras_lookup = {
        "curvature":          curvature,
        "q_balance":          q_balance,
        "cos_x":              cos_x,
        "sin_x":              sin_x,
        "decel_marker":       decel_marker,
        "tke_to_meanke":      tke_to_meanke,
        "viscous_inertia":    viscous_inertia,
        "turb_strain_ratio":  turb_strain_ratio,
        "reverse_flow":       reverse_flow,
        "log_nu_t_ratio":     log_nu_t_ratio,
        "prod_diss_ratio":    prod_diss_ratio,
        "wall_normal_strain": gradU[:, 1, 1],   # dV/dy — normal-stress transport signal
    }
    extra_names = list(stats.get("extra_features", []))

    # 8b. Streamline history features (sl_*) — traced backward from each cell
    sl_names = [n for n in extra_names if n.startswith("sl_")]
    if sl_names:
        sl_feats = compute_streamline_history(cell_xy, U, V, d_wall)
        extras_lookup.update(sl_feats)

    extras = np.stack([extras_lookup[n] for n in extra_names], axis=1) \
             if extra_names else np.zeros((N, 0))

    # 9. Apply training-time transforms
    use_signed_log = bool(stats.get("signed_log_lambdas", False))
    lam_in = signed_log1p(lambdas) if use_signed_log else lambdas.copy()
    wall = np.stack([np.log1p(y_plus), y_delta], axis=1)
    feat_raw = np.concatenate([lam_in, wall, extras], axis=1)

    lo = np.array(stats["lam_lo"]); hi = np.array(stats["lam_hi"])
    feat = np.clip(feat_raw, lo, hi)
    mu = np.array(stats["mu"]); sigma = np.array(stats["sigma"])
    feat_norm = ((feat - mu) / sigma).astype(np.float32)

    # 10. Frobenius-normalise tensor basis
    T_norms = np.sqrt((T.astype(np.float64) ** 2).sum(axis=-1, keepdims=True))
    T_safe = np.where(T_norms > 1e-30, T_norms, 1.0)
    T_norm = (T.astype(np.float64) / T_safe).astype(np.float32)

    aux = dict(
        d_wall=d_wall, y_top=y_top, u_tau_ref=u_tau_ref,
        S_sym=S_raw, O_anti=O_raw, gradU=gradU, k=k, eps=eps, nu_t=nu_t,
        S2_norm=S2_norm, speed=speed,
    )
    return feat_norm, T_norm, aux


def compute_streamline_history(
    cell_xy:   np.ndarray,   # (N, 2)  cell centres
    U:         np.ndarray,   # (N,)    x-velocity
    V:         np.ndarray,   # (N,)    y-velocity
    d_wall:    np.ndarray,   # (N,)    wall distance (unused but kept for API compat)
    dt:        float = 0.05,   # time step in H/U_b units  (matches original script)
    T_max:     float = 50.0,   # max trace length in H/U_b
    max_steps: int   = 1000,
    U_BULK:    float = 0.0196, # bulk velocity (Wu et al. Re=2800 periodic hill)
    H:         float = 1.0,
    batch:     int   = 512,    # cells per vectorised batch
) -> dict:
    """Backward streamline tracer matching add_pehill_streamline_history.py exactly.

    Returns dict with compressed features (signed_log1p already applied):
      sl_reverse_time  = signed_log1p(rev_acc / 5.0)
      sl_min_speed     = signed_log1p(min |vel|/U_bulk along trace)
      sl_dist_to_sep   = signed_log1p(arc-dist to nearest U sign change / 5.0)

    Uses cKDTree nearest-cell lookup, matching the original serial algorithm but
    batched over `batch` starting cells at once for speed.
    """
    N = len(U)
    tree = cKDTree(cell_xy)
    Lx   = cell_xy[:, 0].max() - cell_xy[:, 0].min()   # ~9H
    x_lo = cell_xy[:, 0].min()

    rev_acc  = np.zeros(N, dtype=np.float64)
    min_spd  = np.full(N, np.inf, dtype=np.float64)
    dist_sep = np.full(N, T_max,  dtype=np.float64)

    # Process in batches of `batch` starting cells simultaneously
    for b_start in range(0, N, batch):
        b_end = min(b_start + batch, N)
        B = b_end - b_start

        pos          = cell_xy[b_start:b_end].copy().astype(np.float64)
        rev_b        = np.zeros(B)
        spd_b        = np.full(B, np.inf)
        dsep_b       = np.full(B, T_max)
        last_sign    = np.sign(U[b_start:b_end]).astype(float)
        cum_dist     = np.zeros(B)
        found_sep    = np.zeros(B, dtype=bool)
        active       = np.ones(B, dtype=bool)

        for _ in range(max_steps):
            if not active.any():
                break
            pa = pos[active]
            _, idx = tree.query(pa)
            u_here = U[idx]; v_here = V[idx]

            speed = np.hypot(u_here, v_here) / U_BULK
            # exact step from original: x -= [u,v]/|vel| * dt * U_BULK
            spd_phys = np.maximum(speed * U_BULK, 1e-9)
            step_x = -u_here / spd_phys * dt * U_BULK
            step_y = -v_here / spd_phys * dt * U_BULK

            # reverse-flow accumulation
            rev_b[active] += dt * (u_here < 0.0).astype(float)

            # min speed
            spd_b[active] = np.minimum(spd_b[active], speed)

            # sign-change detection (U positive→negative = entering separation)
            ai = np.where(active)[0]
            new_sign = np.sign(u_here)
            changed = (~found_sep[ai]) & (last_sign[ai] > 0) & (new_sign < 0)
            dsep_b[ai[changed]] = cum_dist[ai[changed]]
            found_sep[ai[changed]] = True
            last_sign[ai] = new_sign

            cum_dist[active] += dt

            pos[active, 0] += step_x
            pos[active, 1] += step_y
            # wrap x periodically
            pos[active, 0] = (pos[active, 0] - x_lo) % Lx + x_lo

            # deactivate particles that left the domain (y < 0 or > y_max)
            y_hi = cell_xy[:, 1].max()
            oob = (pos[active, 1] < 0) | (pos[active, 1] > y_hi * 1.01)
            active[ai[oob]] = False

            if cum_dist.max() >= T_max:
                break

        rev_acc[b_start:b_end]  = rev_b
        min_spd[b_start:b_end]  = np.where(np.isinf(spd_b), 1.0, spd_b)
        dist_sep[b_start:b_end] = dsep_b

    # Apply same compression as original script
    sl_rev  = signed_log1p(rev_acc / 5.0).astype(np.float32)
    sl_spd  = signed_log1p(min_spd).astype(np.float32)
    sl_dist = signed_log1p(dist_sep / 5.0).astype(np.float32)

    return {
        "sl_reverse_time": sl_rev,
        "sl_min_speed":    sl_spd,
        "sl_dist_to_sep":  sl_dist,
    }
