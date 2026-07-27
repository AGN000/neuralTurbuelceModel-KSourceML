"""
Download Wu et al. (2020) periodic hill dataset from GitHub and convert to HDF5.

Source:  https://github.com/xiaoh/para-database-for-PIML
Paper:   Xiao et al., Computers and Fluids 200, 104431 (2020)
         DOI: https://doi.org/10.1016/j.compfluid.2020.104431

Geometry: Periodic hills (2D, extruded 1 cell in z)
Re:       2800 (based on hill height H and bulk velocity)
Cases:    5 hill slope variants — alpha = 0.5, 0.8, 1.0, 1.2, 1.5
          (alpha=1.0 is the classic Frohlich/Mellen periodic hill)

Fields downloaded per case (OpenFOAM ASCII)
--------------------------------------------
  UDNS        mean velocity  (volVectorField)        [m/s]
  TauDNS      Reynolds stress (volSymmTensorField)   [m2/s2]
  k           turbulent kinetic energy               [m2/s2]
  epsilon     dissipation rate                       [m2/s3]

What we compute
---------------
  grad_U[i,j] = dU_i/dx_j  via foamToVTK + finite differences on raw mesh coords
                (approximated by reading cell-centre coordinates from OpenFOAM mesh)
  S_hat, O_hat, 5 invariants, 10 tensor basis T^(n)_ij
  anisotropy b_ij = R_ij/(2k) - delta/3

HDF5 output: /data/TurbuelceModel/data/pehill_alpha{X}.h5
  /meta/    x, y, alpha, Re
  /mean/    U, V, dUdx, dUdy, dVdx, dVdy
  /stress/  R11, R12, R22, R33, k, eps
  /phase1/  features (N,5), nu_t
  /phase2/  lambdas (N,5), T_basis (N,10,9), b_ij (N,9)

Usage
  python download_pehill.py --cases 0p5 0p8 1p0 1p2 1p5
"""

import argparse
import io
import os
import re
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import h5py

RAW_BASE = (
    "https://raw.githubusercontent.com/xiaoh/para-database-for-PIML/master"
    "/pehill-5-cases-OpenFOAM"
)
API_BASE = (
    "https://api.github.com/repos/xiaoh/para-database-for-PIML/contents"
    "/pehill-5-cases-OpenFOAM"
)
OUTDIR   = Path("/data/TurbuelceModel/data")

CASES = {
    "0p5": 0.5, "0p8": 0.8, "1p0": 1.0, "1p2": 1.2, "1p5": 1.5
}
RE = 2800   # Reynolds number for this dataset

#  OpenFOAM ASCII field reader 

def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_foam_vector_field(text: str) -> np.ndarray:
    m = re.search(r"nonuniform List<vector>\s*\n\s*(\d+)\s*\n\(", text)
    if not m:
        raise ValueError("Could not find vector field data")
    N = int(m.group(1))
    start = m.end()
    end   = text.index("\n)", start)
    block = text[start:end]
    rows  = re.findall(r"\(([^)]+)\)", block)
    arr   = np.array([[float(v) for v in r.split()] for r in rows if len(r.split()) == 3])
    assert arr.shape == (N, 3), f"Expected ({N},3), got {arr.shape}"
    return arr


def parse_foam_symm_tensor_field(text: str) -> np.ndarray:
    m = re.search(r"nonuniform List<symmTensor>\s*\n\s*(\d+)\s*\n\(", text)
    if not m:
        raise ValueError("Could not find symmTensor field data")
    N = int(m.group(1))
    start = m.end()
    end   = text.index("\n)", start)
    block = text[start:end]
    # Capture everything inside each pair of parens and split
    rows = re.findall(r"\(([^)]+)\)", block)
    arr  = np.array([[float(v) for v in r.split()] for r in rows])
    assert arr.shape == (N, 6), f"Expected ({N},6), got {arr.shape}"
    return arr


def parse_foam_scalar_field(text: str, field_type: str = "scalar") -> np.ndarray:
    m = re.search(
        rf"nonuniform List<{field_type}>\s*\n\s*(\d+)\s*\n\(", text
    )
    if not m:
        raise ValueError(f"Could not find {field_type} field data")
    N = int(m.group(1))
    start = m.end()
    end   = text.index("\n)", start)
    vals  = list(map(float, text[start:end].split()))
    arr   = np.array(vals)
    assert arr.shape == (N,), f"Expected ({N},), got {arr.shape}"
    return arr


def parse_foam_points(text: str) -> np.ndarray:
    m = re.search(r"(\d+)\s*\n\(", text)
    if not m:
        raise ValueError("Could not find points data")
    N = int(m.group(1))
    start = m.end()
    end   = text.index("\n)", start)
    block = text[start:end]
    rows  = re.findall(r"\(([^)]+)\)", block)
    arr   = np.array([[float(v) for v in r.split()] for r in rows if len(r.split()) == 3])
    return arr


def parse_foam_faces(text: str) -> list[list[int]]:
    m = re.search(r"(\d+)\s*\n\(", text)
    N = int(m.group(1))
    start = m.end()
    # Extract n(v0 v1 ... vn-1) for each face
    faces = re.findall(r"\d+\(([^)]+)\)", text[start:start + N * 50 + 10000])
    return [[int(x) for x in f.split()] for f in faces[:N]]


def compute_cell_centres(points_url: str, faces_url: str,
                          owner_url: str, neighbour_url: str,
                          N_cells: int) -> np.ndarray:
    pts_text = _fetch(points_url)
    pts = parse_foam_points(pts_text)

    fac_text = _fetch(faces_url)
    faces = parse_foam_faces(fac_text)

    own_text = _fetch(owner_url)
    m = re.search(r"(\d+)\s*\n\(", own_text)
    N_faces = int(m.group(1))
    start = m.end()
    end   = own_text.index("\n)", start)
    owners = list(map(int, own_text[start:end].split()))

    # Accumulate face centres into owning cell
    cell_sum   = np.zeros((N_cells, 3))
    cell_count = np.zeros(N_cells, dtype=int)
    for fi, (verts, owner) in enumerate(zip(faces, owners)):
        fc = pts[verts].mean(axis=0)
        cell_sum[owner]   += fc
        cell_count[owner] += 1

    # Avoid division by zero
    valid = cell_count > 0
    centres = np.where(valid[:, None], cell_sum / np.maximum(cell_count[:, None], 1), 0)
    return centres  # (N_cells, 3)


# Tensor algebra helpers

def sym(A):
    return 0.5 * (A + A.transpose(0, 2, 1))

def antisym(A):
    return 0.5 * (A - A.transpose(0, 2, 1))

def mat_mul(A, B):
    return np.einsum("nij,njk->nik", A, B)

def trace(A):
    return np.einsum("nii->n", A)

def identity_like(A):
    N = A.shape[0]
    I = np.zeros_like(A)
    I[:, 0, 0] = I[:, 1, 1] = I[:, 2, 2] = 1.0
    return I


def compute_grad_U_from_cells(xy: np.ndarray, UV: np.ndarray) -> np.ndarray:

    from scipy.spatial import KDTree

    N = xy.shape[0]
    tree = KDTree(xy)
    k_nn = min(9, N)   
    _, idx = tree.query(xy, k=k_nn)

    grad_U = np.zeros((N, 3, 3))

    for i in range(N):
        nbrs = idx[i]       # (k,)
        dx   = xy[nbrs] - xy[i]    # (k, 2)
        dU   = UV[nbrs] - UV[i]    # (k, 2)  Ux, Vx onl

        A = dx  
        try:
            G, _, _, _ = np.linalg.lstsq(A, dU, rcond=None)  # (2, 2)
            # G[j, i] = dU_i / dx_j (note transpose)
            grad_U[i, 0, 0] = G[0, 0]  # dU/dx
            grad_U[i, 0, 1] = G[1, 0]  # dU/dy
            grad_U[i, 1, 0] = G[0, 1]  # dV/dx
            grad_U[i, 1, 1] = G[1, 1]  # dV/dy
        except Exception:
            pass

    return grad_U


def compute_scalar_invariants(S_hat: np.ndarray, O_hat: np.ndarray) -> np.ndarray:
    S2 = mat_mul(S_hat, S_hat)
    S3 = mat_mul(S2, S_hat)
    O2 = mat_mul(O_hat, O_hat)
    O2S  = mat_mul(O2, S_hat)
    O2S2 = mat_mul(O2, S2)
    return np.stack([trace(S2), trace(O2), trace(S3),
                     trace(O2S), trace(O2S2)], axis=1)


def compute_tensor_basis(S_hat: np.ndarray, O_hat: np.ndarray) -> np.ndarray:
    I   = identity_like(S_hat)
    S2  = mat_mul(S_hat, S_hat)
    O2  = mat_mul(O_hat, O_hat)
    SO  = mat_mul(S_hat, O_hat)
    OS  = mat_mul(O_hat, S_hat)
    OS2 = mat_mul(O_hat, S2)
    S2O = mat_mul(S2, O_hat)
    SO2 = mat_mul(S_hat, O2)
    O2S = mat_mul(O2, S_hat)
    def _e(v): return v[:, None, None] * I
    T1 = S_hat
    T2 = SO - OS
    T3 = S2 - _e(trace(S2) / 3)
    T4 = O2 - _e(trace(O2) / 3)
    T5 = OS2 - S2O
    T6 = O2S + SO2 - _e(2 * trace(SO2) / 3)
    T7 = mat_mul(mat_mul(O_hat, S_hat), O2) - mat_mul(mat_mul(O2, S_hat), O_hat)
    T8 = mat_mul(mat_mul(S_hat, O2), S2) - mat_mul(mat_mul(S2, O2), S_hat)
    T9 = mat_mul(O2, S2) + mat_mul(S2, O2) - _e(2 * trace(mat_mul(S2, O2)) / 3)
    T10= mat_mul(mat_mul(O_hat, S2), O2) - mat_mul(mat_mul(O2, S2), O_hat)
    return np.stack([T1,T2,T3,T4,T5,T6,T7,T8,T9,T10], axis=1)




def process_case(case_tag: str, alpha: float, outdir: Path):
    h5path = outdir / f"pehill_alpha{case_tag}.h5"
    if h5path.exists() and not getattr(process_case, '_force', False):
        print(f"  [skip] {h5path.name}")
        return h5path

    base = f"{RAW_BASE}/case_{case_tag}/0"
    mesh_base = f"{RAW_BASE}/case_{case_tag}/constant/polyMesh"

    print(f"  Downloading UDNS ...")
    UDNS_text = _fetch(f"{base}/UDNS")
    UV = parse_foam_vector_field(UDNS_text)   # (N, 3)
    N  = UV.shape[0]

    print(f"  Downloading TauDNS ...")
    tau_text  = _fetch(f"{base}/TauDNS")
    Tau = parse_foam_symm_tensor_field(tau_text)  # (N, 6): R11 R12 R13 R22 R23 R33

    print(f"  Downloading nut (DNS eddy viscosity) ...")
    nut_text = _fetch(f"{base}/nut")
    nut = parse_foam_scalar_field(nut_text)   # nu_t from DNS  [m2/s]

    print(f"  Downloading mesh (points, faces, owner) ...")
    pts_text   = _fetch(f"{mesh_base}/points")
    faces_text = _fetch(f"{mesh_base}/faces")
    owner_text = _fetch(f"{mesh_base}/owner")

    pts    = parse_foam_points(pts_text)
    faces  = parse_foam_faces(faces_text)
    # owners
    m = re.search(r"(\d+)\s*\n\(", owner_text)
    N_faces = int(m.group(1))
    start = m.end()
    end   = owner_text.index("\n)", start)
    owners = list(map(int, owner_text[start:end].split()))

    # Cell centres
    cell_sum   = np.zeros((N, 3))
    cell_count = np.zeros(N, dtype=int)
    for fi, owner in enumerate(owners):
        if owner < N:
            fc = pts[faces[fi]].mean(axis=0)
            cell_sum[owner]   += fc
            cell_count[owner] += 1
    valid  = cell_count > 0
    centres = np.where(valid[:, None],
                       cell_sum / np.maximum(cell_count[:, None], 1), 0.0)
    xy = centres[:, :2]  # (N, 2)

    print(f"  Computing velocity gradients ({N} cells) ...")
    from scipy.spatial import KDTree
    tree = KDTree(xy)
    k_nn = min(12, N)
    _, idx = tree.query(xy, k=k_nn)
    grad_U_raw = np.zeros((N, 3, 3))
    UV2 = UV[:, :2]
    for i in range(N):
        nbrs = idx[i]
        dx   = xy[nbrs] - xy[i]
        dU   = UV2[nbrs] - UV2[i]
        G, _, _, _ = np.linalg.lstsq(dx, dU, rcond=None)
        grad_U_raw[i, 0, 0] = G[0, 0]
        grad_U_raw[i, 0, 1] = G[1, 0]
        grad_U_raw[i, 1, 0] = G[0, 1]
        grad_U_raw[i, 1, 1] = G[1, 1]

    S_raw, O_raw = sym(grad_U_raw), antisym(grad_U_raw)

    # Turbulence quantities from DNS 
    R11 = Tau[:, 0]; R12b = Tau[:, 1]; R22 = Tau[:, 3]; R33 = Tau[:, 5]
    k   = 0.5 * (R11 + R22 + R33)      
    Cmu = 0.09
    eps = np.where(nut > 1e-10, Cmu * k**2 / nut, 0.0)  # Cmu k^2 / nu_t

    # Non-dimensionalise tensors
    # Clip k/eps to physically reasonable range (prevents T_basis blow-up near walls)
    k_over_eps_1d = np.clip(np.where(eps > 1e-10, k / eps, 0.0), 0.0, 100.0)
    k_over_eps    = k_over_eps_1d[:, None, None]            # (N,1,1)
    S_hat = k_over_eps * S_raw
    O_hat = k_over_eps * O_raw

    # Phase 1 features
    SijSij = 0.5 * trace(mat_mul(S_raw, S_raw))
    OijOij = -0.5 * trace(mat_mul(O_raw, O_raw))   # negative sign: trace(O·O) ≤ 0
    S_star  = k_over_eps_1d * np.sqrt(np.maximum(2 * SijSij, 0))
    O_star  = k_over_eps_1d * np.sqrt(np.maximum(2 * OijOij, 0))
    Re_t    = np.clip(np.where(eps > 1e-10, k**2 / (1.5e-5 * eps), 0.0), 0, 1e6)
    feat1   = np.stack([S_star, O_star, Re_t, centres[:, 1], centres[:, 1]], axis=1)
    nu_t_out = np.clip(nut, 0, None)   # DNS nut directly

    # Phase 2 — tensor basis and anisotropy
    lambdas = compute_scalar_invariants(S_hat, O_hat)
    T_basis = compute_tensor_basis(S_hat, O_hat)

    # Clip lambda outliers caused by near-wall/separated-region singularities.
    # Use 1st–99th percentile of interior points only (where k > 0).
    interior = k > 1e-12
    for j in range(lambdas.shape[1]):
        lo = np.percentile(lambdas[interior, j], 1)
        hi = np.percentile(lambdas[interior, j], 99)
        lambdas[:, j] = np.clip(lambdas[:, j], lo, hi)
    T_basis = np.clip(T_basis, -1e4, 1e4)  # hard cap on basis tensors

    safe_k = np.where(k > 1e-30, k, np.nan)
    b = np.zeros((N, 3, 3))
    b[:, 0, 0] = np.nan_to_num(R11  / (2*safe_k) - 1/3, nan=0.0)
    b[:, 0, 1] = np.nan_to_num(R12b / (2*safe_k),       nan=0.0)
    b[:, 1, 0] = b[:, 0, 1]
    b[:, 1, 1] = np.nan_to_num(R22  / (2*safe_k) - 1/3, nan=0.0)
    b[:, 2, 2] = np.nan_to_num(R33  / (2*safe_k) - 1/3, nan=0.0)

    print(f"  Saving {h5path} ({N} cells) ...")
    with h5py.File(h5path, "w") as f:
        f.attrs["alpha"] = alpha
        f.attrs["Re"]    = float(RE)
        f.attrs["N"]     = N
        f.attrs["source"] = "Wu/Xiao para-database-for-PIML"

        meta = f.create_group("meta")
        meta.create_dataset("x",      data=xy[:, 0].astype(np.float32))
        meta.create_dataset("y",      data=xy[:, 1].astype(np.float32))

        mean = f.create_group("mean")
        mean.create_dataset("U",  data=UV[:, 0].astype(np.float32))
        mean.create_dataset("V",  data=UV[:, 1].astype(np.float32))
        for name, val in [("dUdx", grad_U_raw[:,0,0]),
                          ("dUdy", grad_U_raw[:,0,1]),
                          ("dVdx", grad_U_raw[:,1,0]),
                          ("dVdy", grad_U_raw[:,1,1])]:
            mean.create_dataset(name, data=val.astype(np.float32))

        st = f.create_group("stress")
        st.create_dataset("R11", data=R11.astype(np.float32))
        st.create_dataset("R12", data=R12b.astype(np.float32))
        st.create_dataset("R22", data=R22.astype(np.float32))
        st.create_dataset("R33", data=R33.astype(np.float32))
        st.create_dataset("k",   data=k.astype(np.float32))
        st.create_dataset("eps", data=eps.astype(np.float32))

        p1 = f.create_group("phase1")
        p1.create_dataset("features",  data=feat1.astype(np.float32))
        p1.create_dataset("nu_t",      data=nu_t_out.astype(np.float32))

        p2 = f.create_group("phase2")
        p2.create_dataset("lambdas",   data=lambdas.astype(np.float32))
        p2.create_dataset("T_basis",   data=T_basis.reshape(N,10,9).astype(np.float32))
        p2.create_dataset("b_ij",      data=b.reshape(N,9).astype(np.float32))

    print(f"  Done — nu_t max={nu_t_out.max():.4e}  k max={k.max():.4e}")
    return h5path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=["0p5","0p8","1p0","1p2","1p5"],
                   choices=list(CASES.keys()))
    p.add_argument("--outdir", type=Path, default=OUTDIR)
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    # Scipy needed for KDTree gradient
    try:
        import scipy
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy", "-q"])
    for tag in args.cases:
        alpha = CASES[tag]
        print(f"\nCase alpha={alpha} ({tag})")
        try:
            process_case(tag, alpha, args.outdir)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")
    print("\nAll done.")


if __name__ == "__main__":
    main()
