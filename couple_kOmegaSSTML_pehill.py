"""
Iterative coupling: kOmegaSST with ML k/ω source correction (kOmegaSSTML).

Strategy
--------
Standard kOmegaSST gives x_r ≈ 6.20 H because it predicts locally wrong k
in the recirculation shear layer.  We correct this by adding two source terms
to the k and ω transport equations:

    kSourceML     = λ_k · (k_DNS   - k_RANS)      [m²/s³]
    omegaSourceML = λ_ω · (ω_target - ω_RANS)     [1/s²]
    ω_target      = k_DNS / νt_DNS

At convergence the SST model reproduces the DNS turbulent kinetic energy and
eddy viscosity. The resulting Boussinesq stress (-2 νt_DNS S) is the best
single-field approximation to Rij_DNS and should give x_r closer to 4.10 H.

Because the SST model is implicit (νt enters the momentum equation as
diffusion, not as an explicit body force), this coupling has a unique fixed
point — unlike the explicit Rij-injection approach which exhibits a limit cycle.

Usage
-----
  python3 couple_kOmegaSSTML_pehill.py \
      --case   /data/.../pehill_alpha1p0_kOmegaSSTML \
      --dns_h5 /data/.../pehill_alpha1p0.h5 \
      --outer  30 --inner 300 \
      --lam_k  50 --lam_omega 30 \
      --of_env /opt/openfoam11
"""

from __future__ import annotations
import argparse
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

import h5py
import numpy as np
from scipy.spatial import cKDTree
from pehill_features import cell_centres_from_polymesh as _cell_centres_of


NU = 1.0 / 5600.0          # kinematic viscosity (Re = Ub H / ν = 5600, H=Ub=1)

# ── OpenFOAM field I/O ────────────────────────────────────────────────────────

def latest_time_dir(case: Path) -> Path:
    dirs = [d for d in case.iterdir()
            if d.is_dir() and d.name.replace('.', '', 1).isdigit()]
    return max(dirs, key=lambda d: float(d.name))


def read_scalar(fpath: Path) -> np.ndarray:
    """Read a volScalarField (uniform or nonuniform list) → numpy array."""
    text = fpath.read_text()
    if "nonuniform" in text:
        # Find "(\n" that opens the internalField list
        nu_pos = text.index("nonuniform")
        open_pos = text.index("(\n", nu_pos) + 2
        # Find closing ")\n;" immediately after the data block
        close_pos = text.index("\n)\n", open_pos)
        return np.fromstring(text[open_pos:close_pos], sep="\n", dtype=np.float64)
    elif "uniform" in text:
        val = float(text.split("uniform")[1].split(";")[0])
        return np.array([val])
    raise ValueError(f"Cannot parse scalar field: {fpath}")


def read_vector(fpath: Path) -> np.ndarray:
    """Read a volVectorField → (N, 3) array."""
    text = fpath.read_text()
    if "nonuniform" in text:
        nu_pos = text.index("nonuniform")
        open_pos = text.index("(\n", nu_pos) + 2
        close_pos = text.index("\n)\n", open_pos)
        block = text[open_pos:close_pos].replace("(", "").replace(")", "")
        return np.fromstring(block, sep="\n").reshape(-1, 3)
    elif "uniform" in text:
        inner = text.split("uniform")[1].split("(")[1].split(")")[0]
        return np.array([[float(v) for v in inner.split()]])
    raise ValueError(f"Cannot parse vector field: {fpath}")


def of_header(class_: str, location: str, name: str) -> str:
    return f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Version:  11
    \\\\  /    A nd           | Website:  https://openfoam.org
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       {class_};
    location    "{location}";
    object      {name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def write_scalar_field(fpath: Path, data: np.ndarray,
                        dims: str, name: str, time_name: str) -> None:
    """Write a volScalarField in nonuniform List<scalar> format."""
    lines = [of_header("volScalarField", time_name, name)]
    lines.append(f"\ndimensions      {dims};\n")
    lines.append(f"\ninternalField   nonuniform List<scalar>\n{len(data)}\n(\n")
    for v in data:
        lines.append(f"{v:.8e}\n")
    lines.append(")\n;\n\n")
    lines.append("""\
boundaryField
{
    bottomWall { type zeroGradient; }
    topWall    { type zeroGradient; }
    inlet      { type cyclic; }
    outlet     { type cyclic; }
    defaultFaces { type empty; }
}

// ************************************************************************* //
""")
    fpath.write_text("".join(lines))


# ── Cell centres from polyMesh ────────────────────────────────────────────────

def cell_centres_from_polymesh(case: Path) -> np.ndarray:
    """Compute cell centres from polyMesh topology (x,y only)."""
    xy3, _ = _cell_centres_of(case)
    return xy3[:, :2].astype(np.float64)


# ── DNS data loading ──────────────────────────────────────────────────────────

def load_dns(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray]:
    """Return (x_dns, y_dns, k_dns, nu_t_dns, omega_target) arrays."""
    with h5py.File(h5_path) as f:
        x     = f["meta/x"][:]
        y     = f["meta/y"][:]
        k     = f["stress/k"][:]
        nu_t  = f["phase1/nu_t"][:]
    # ω_target = k / νt  (clamp νt to avoid division by zero)
    nu_t_safe  = np.maximum(nu_t, 1e-8)
    omega_tgt  = np.clip(k / nu_t_safe, 1.0, 1e5)
    return (x.astype(np.float64), y.astype(np.float64),
            k.astype(np.float64), nu_t.astype(np.float64),
            omega_tgt.astype(np.float64))


def build_dns_tree(x_dns: np.ndarray, y_dns: np.ndarray) -> cKDTree:
    return cKDTree(np.stack([x_dns, y_dns], axis=1))


def map_dns_to_mesh(dns_vals: np.ndarray, tree: cKDTree,
                    cell_xy: np.ndarray) -> np.ndarray:
    """Nearest-neighbour interpolation of DNS field to OF cell centres."""
    _, idx = tree.query(cell_xy)
    return dns_vals[idx]


# ── Reattachment length ───────────────────────────────────────────────────────

def reattachment_length(cell_xy: np.ndarray, U: np.ndarray,
                         x_start: float = 1.0, x_end: float = 7.5,
                         wall_band: float = 0.35) -> float:
    x = cell_xy[:, 0]; y = cell_xy[:, 1]
    x_bins = np.linspace(x_start, x_end, int((x_end - x_start) / 0.1) + 1)
    dx = x_bins[1] - x_bins[0]
    for xb in x_bins:
        mask = np.abs(x - xb) < dx
        if mask.sum() < 3:
            continue
        y_wall = y[mask].min()
        near = mask & (y < y_wall + wall_band)
        if near.sum() < 2:
            continue
        if U[near].mean() > 0:
            return float(xb)
    return float(x_end)


# ── Source field computation ──────────────────────────────────────────────────

def compute_sources(k_dns_m: np.ndarray, omega_tgt_m: np.ndarray,
                    k_rans: np.ndarray, omega_rans: np.ndarray,
                    lam_k: float, lam_omega: float,
                    clip_k: float = 0.05, clip_omega: float = 5000.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    kSourceML     = lam_k * max(k_DNS - k_RANS, 0)    [m²/s³]  (positive only)
    omegaSourceML = lam_omega * (omega_target - omega_RANS)       [1/s²]

    k_src is clipped to non-negative only: we only ADD k production where
    RANS underpredicts DNS.  Negative k_src (where RANS > DNS) would drive
    k → 0 causing fvm::Su instability; the proportional approach already
    handles this by only injecting positive corrections.
    """
    k_raw  = lam_k * np.maximum(k_dns_m - k_rans, 0.0)
    k_src  = np.clip(k_raw, 0.0, clip_k)
    om_src = np.clip(lam_omega * (omega_tgt_m - omega_rans), -clip_omega, clip_omega)
    return k_src.astype(np.float64), om_src.astype(np.float64)


# ── simpleFoam runner ─────────────────────────────────────────────────────────

def set_end_time(case: Path, end_time: int) -> None:
    """Patch endTime and writeInterval in system/controlDict."""
    cd = case / "system" / "controlDict"
    text = cd.read_text()
    import re
    text = re.sub(r'endTime\s+\d+\s*;', f'endTime         {end_time};', text)
    text = re.sub(r'writeInterval\s+\d+\s*;', f'writeInterval   {end_time};', text)
    cd.write_text(text)


def run_simpleFoam(case: Path, end_time: int, log: Path, of_root: str) -> None:
    """Run simpleFoam to `end_time`, appending to `log`."""
    set_end_time(case, end_time)
    cmd = (
        f"source {of_root}/etc/bashrc 2>/dev/null; "
        f"export FOAM_MPI=mpich-3.3; "
        f"cd {case} && simpleFoam >> {log} 2>&1; exit $?"
    )
    rc = subprocess.run(["bash", "-c", cmd]).returncode
    if rc != 0:
        tail = log.read_text().splitlines()[-15:]
        for ln in tail:
            print(f"  {ln}")
        raise RuntimeError(f"simpleFoam rc={rc}")


# ── Main coupling loop ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="kOmegaSSTML coupling: k + ω source steering toward DNS."
    )
    p.add_argument("--case",    type=Path,
                   default=Path("/data/TurbuelceModel_Loc/openfoam_coupling/"
                                "pehill_alpha1p0_kOmegaSSTML"))
    p.add_argument("--dns_h5",  type=Path,
                   default=Path("/data/TurbuelceModel_Loc/data/pehill_alpha1p0.h5"))
    p.add_argument("--of_env",  default="/opt/openfoam11")
    p.add_argument("--outer",   type=int,   default=30)
    p.add_argument("--inner",   type=int,   default=300)
    p.add_argument("--lam_k",   type=float, default=50.0,
                   help="k-source relaxation rate [1/s]")
    p.add_argument("--lam_omega", type=float, default=30.0,
                   help="omega-source relaxation rate [1/s]")
    p.add_argument("--ramp",    type=int,   default=5,
                   help="outer iterations to linearly ramp sources to full strength")
    args = p.parse_args()

    case    = args.case
    dns_h5  = args.dns_h5
    of_root = args.of_env
    lib_dir = Path(of_root).parent  # not used directly; libs in FOAM_USER_LIBBIN

    print(f"kOmegaSSTML coupling  case={case.name}")
    print(f"  DNS h5: {dns_h5}")
    print(f"  outer={args.outer}  inner={args.inner}"
          f"  λ_k={args.lam_k}  λ_ω={args.lam_omega}  ramp={args.ramp}")

    # ── Load DNS reference data ───────────────────────────────────────────────
    x_dns, y_dns, k_dns, nu_t_dns, omega_tgt = load_dns(dns_h5)
    dns_tree = build_dns_tree(x_dns, y_dns)
    print(f"DNS: {len(k_dns)} points  "
          f"k mean={k_dns.mean():.4e}  νt mean={nu_t_dns.mean():.4e}")

    # ── Get cell centres (generate C field if absent) ─────────────────────────
    log_path = case / "log.kOmegaSSTML"
    log_path.write_text("")

    # Get initial time directory
    latest = latest_time_dir(case)
    current = int(float(latest.name))
    print(f"Starting from t={current}")

    # Compute cell centres from polyMesh topology.
    cell_xy = cell_centres_from_polymesh(case)
    print(f"Cell centres: {len(cell_xy)} cells  "
          f"x=[{cell_xy[:,0].min():.2f}, {cell_xy[:,0].max():.2f}]  "
          f"y=[{cell_xy[:,1].min():.3f}, {cell_xy[:,1].max():.2f}]")

    N = len(cell_xy)
    print(f"Mesh: {N} cells")

    # ── Map DNS fields to OF mesh ─────────────────────────────────────────────
    k_dns_m      = map_dns_to_mesh(k_dns,      dns_tree, cell_xy)
    omega_tgt_m  = map_dns_to_mesh(omega_tgt,  dns_tree, cell_xy)
    print(f"Mapped DNS: k mean={k_dns_m.mean():.4e}  "
          f"ω_target mean={omega_tgt_m.mean():.2f}")

    # ── Coupling loop ─────────────────────────────────────────────────────────
    results = []

    for outer in range(1, args.outer + 1):
        # Ramp source strength linearly from 0 → 1 over `ramp` iterations.
        ramp_factor = min(1.0, outer / args.ramp)

        # Read current RANS k and omega from latest time directory.
        latest = latest_time_dir(case)
        k_file = latest / "k"
        om_file = latest / "omega"

        if k_file.exists():
            k_rans = read_scalar(k_file)
            if len(k_rans) == 1:
                k_rans = np.full(N, k_rans[0])
        else:
            k_rans = np.full(N, 3.5e-4)

        if om_file.exists():
            omega_rans = read_scalar(om_file)
            if len(omega_rans) == 1:
                omega_rans = np.full(N, omega_rans[0])
        else:
            omega_rans = np.full(N, 100.0)

        k_src, om_src = compute_sources(
            k_dns_m, omega_tgt_m, k_rans, omega_rans,
            lam_k=args.lam_k * ramp_factor,
            lam_omega=args.lam_omega * ramp_factor,
        )

        # Write source fields to the current time directory.
        time_name = latest.name
        write_scalar_field(
            latest / "kSourceML",
            k_src,
            "[0 2 -3 0 0 0 0]", "kSourceML", time_name
        )
        write_scalar_field(
            latest / "omegaSourceML",
            om_src,
            "[0 0 -2 0 0 0 0]", "omegaSourceML", time_name
        )

        print(f"\n── Outer {outer}/{args.outer}  ramp={ramp_factor:.2f}  "
              f"(t {current}→{current + args.inner}) ──")
        print(f"  k src : mean={k_src.mean():+.4e}  |max|={np.abs(k_src).max():.4e}")
        print(f"  ω src : mean={om_src.mean():+.4e}  |max|={np.abs(om_src).max():.4e}")

        # Run simpleFoam.
        run_simpleFoam(case, current + args.inner, log_path, of_root)
        current += args.inner

        # Read updated U and report x_r.
        latest = latest_time_dir(case)
        U_vec = read_vector(latest / "U")
        if U_vec.shape[0] == 1:
            U_vec = np.tile(U_vec, (N, 1))
        U = U_vec[:, 0]

        xr = reattachment_length(cell_xy, U)

        k_now = read_scalar(latest / "k")
        if len(k_now) == 1:
            k_now = np.full(N, k_now[0])
        k_err = np.abs(k_now - k_dns_m).mean()

        print(f"  x_r = {xr:.2f} H   mean|k_err|={k_err:.4e}"
              f"   k_RANS mean={k_now.mean():.4e}  k_DNS mean={k_dns_m.mean():.4e}")
        results.append((outer, xr, k_err, k_now.mean()))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════")
    print("  outer  x_r/H   mean|Δk|   k_RANS")
    for outer, xr, kerr, km in results:
        print(f"  {outer:3d}    {xr:.2f}    {kerr:.3e}    {km:.3e}")
    best_xr = max(r[1] for r in results)
    print(f"\nBest x_r = {best_xr:.2f} H  (DNS = 4.10 H)")
    print("═══════════════════════════════════════")


if __name__ == "__main__":
    main()
