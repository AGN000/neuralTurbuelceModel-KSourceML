"""
Field-inversion sweep: find the Rij scale factor s* such that
RANS(s* · Rij_DNS) has a stable fixed point at x_r ≈ 4.10H.

Theory (from Jacobian analysis):
  - RANS(s=1.0, Rij_DNS) → x_r ≈ 1.68H  (too short, too much stress)
  - RANS(s=0.0, no Rij)  → x_r ≈ 6–8H  (laminar, no stress)
  - Target x_r = 4.10H  → expect s* ∈ (0.1, 0.5)

Workflow per s value:
  1. Clone the base laminar case (pehill_alpha1p0_Rij_pos) to a fresh dir.
  2. Run couple_Rij_NODE_pehill.py in oracle mode with --rij_scale s.
  3. Parse the log for the converged x_r.
  4. Report x_r(s) table and interpolate s*.

Usage:
  python3 field_inversion_sweep.py [--scales 0.05 0.1 0.2 0.3 0.5 0.7 1.0]
                                   [--outer 25] [--inner 200]
                                   [--of_env /opt/openfoam11]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

COUPLING_DIR = Path(__file__).resolve().parent
ROOT         = COUPLING_DIR.parent
PYTHON       = sys.executable

BASE_CASE    = COUPLING_DIR / "pehill_alpha1p0_Rij_pos"
DNS_H5       = ROOT / "data" / "pehill_alpha1p0.h5"
POS_CKPT     = ROOT / "phase2_stress" / "checkpoints_pos_rij" / "best_pos_rij.pt"
OF_ENV_DEFAULT = "/opt/openfoam11"

COUPLE_SCRIPT = COUPLING_DIR / "couple_Rij_NODE_pehill.py"


def clone_fresh_case(base: Path, dst: Path) -> None:
    """Clone base case keeping only t=0 initial conditions."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(base, dst)
    # Remove all time dirs except 0
    for d in dst.iterdir():
        if d.is_dir() and d.name.isdigit() and d.name != "0":
            shutil.rmtree(d)
    # Remove tbnn_dumps if present
    dumps = dst / "tbnn_dumps"
    if dumps.exists():
        shutil.rmtree(dumps)
    # Clean any existing logs
    for f in dst.glob("log.*"):
        f.unlink()


def run_coupling(case: Path, scale: float, outer: int, inner: int,
                 of_env: str, log_file: Path) -> None:
    cmd = [
        PYTHON, str(COUPLE_SCRIPT),
        "--case",       str(case),
        "--mode",       "oracle",
        "--model_type", "pos",
        "--dns_h5",     str(DNS_H5),
        "--rij_scale",  str(scale),
        "--outer",      str(outer),
        "--inner",      str(inner),
        "--relax",      "0.3",
        "--nut_floor",  "5.0",
        "--of_env",     of_env,
    ]
    with open(log_file, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=fh).returncode
    if rc != 0:
        print(f"  [WARNING] coupling exited with rc={rc} — check {log_file}")


def parse_xr_from_log(log_file: Path) -> list[float]:
    """Extract x_r/H values from coupling log (one per outer iteration)."""
    xr_values = []
    pattern = re.compile(r"x_r/H\s*=\s*([0-9.]+)H")
    for line in log_file.read_text().splitlines():
        m = pattern.search(line)
        if m:
            xr_values.append(float(m.group(1)))
    return xr_values


def interpolate_s_star(scales: list[float], xr_final: list[float],
                       target: float = 4.10) -> float | None:
    """Linear interpolation to find s* where x_r(s*) = target."""
    for i in range(len(scales) - 1):
        x0, x1 = xr_final[i], xr_final[i + 1]
        if (x0 - target) * (x1 - target) <= 0:
            s0, s1 = scales[i], scales[i + 1]
            frac = (target - x0) / (x1 - x0)
            return float(s0 + frac * (s1 - s0))
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00])
    p.add_argument("--outer",  type=int, default=25)
    p.add_argument("--inner",  type=int, default=200)
    p.add_argument("--of_env", type=str, default=OF_ENV_DEFAULT)
    p.add_argument("--target_xr", type=float, default=4.10,
                   help="Target reattachment length (H) for interpolation")
    args = p.parse_args()

    sweep_dir = COUPLING_DIR / "fi_sweep"
    sweep_dir.mkdir(exist_ok=True)

    results: list[tuple[float, list[float]]] = []

    print(f"\n{'='*60}")
    print(f"Field-Inversion Sweep  (target x_r = {args.target_xr}H)")
    print(f"Scales: {args.scales}")
    print(f"Outer={args.outer}  Inner={args.inner}  [PARALLEL]")
    print(f"{'='*60}\n")

    # ── Phase 1: clone all cases and launch all subprocesses in parallel ──────
    procs: list[tuple[float, Path, Path, subprocess.Popen]] = []
    for s in args.scales:
        case_name = f"fi_s{s:.2f}".replace(".", "p")
        case_dir  = sweep_dir / case_name
        log_file  = sweep_dir / f"log_{case_name}.txt"

        print(f"Cloning & launching s={s:.2f} → {case_dir.name}")
        clone_fresh_case(BASE_CASE, case_dir)

        cmd = [
            PYTHON, str(COUPLE_SCRIPT),
            "--case",       str(case_dir),
            "--mode",       "oracle",
            "--model_type", "pos",
            "--dns_h5",     str(DNS_H5),
            "--rij_scale",  str(s),
            "--outer",      str(args.outer),
            "--inner",      str(args.inner),
            "--relax",      "0.3",
            "--nut_floor",  "5.0",
            "--of_env",     args.of_env,
        ]
        fh = open(log_file, "w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=fh)
        procs.append((s, case_dir, log_file, proc))

    print(f"\nAll {len(procs)} jobs launched. Waiting for completion…\n")

    # ── Phase 2: wait for all and collect results ─────────────────────────────
    for s, case_dir, log_file, proc in procs:
        rc = proc.wait()
        xr_traj = parse_xr_from_log(log_file)
        if xr_traj:
            print(f"  s={s:.2f}  rc={rc}  x_r traj: "
                  f"{[f'{x:.2f}H' for x in xr_traj]}  "
                  f"final={xr_traj[-1]:.2f}H  peak={max(xr_traj):.2f}H")
        else:
            print(f"  s={s:.2f}  rc={rc}  WARNING: no x_r in log — check {log_file}")
        results.append((s, xr_traj))

    # Summary table
    print(f"\n{'='*60}")
    print("Summary:  s   |  peak x_r/H  |  final x_r/H")
    print("-" * 46)
    scales_done, xr_finals = [], []
    for s, traj in results:
        if traj:
            pk = max(traj)
            fn = traj[-1]
            print(f"  {s:.2f}   |   {pk:.2f}H       |   {fn:.2f}H")
            scales_done.append(s)
            xr_finals.append(fn)
        else:
            print(f"  {s:.2f}   |   —            |   —  (failed)")

    # Interpolate s*
    if len(scales_done) >= 2:
        s_star = interpolate_s_star(scales_done, xr_finals, target=args.target_xr)
        if s_star is not None:
            print(f"\n  → Interpolated s* ≈ {s_star:.3f}  (x_r = {args.target_xr}H)")
            print(f"    Train pos-model on {s_star:.3f} × Rij_DNS to get RANS-compatible stress.")
        else:
            print(f"\n  → Target {args.target_xr}H not bracketed by sweep range.")
            print(f"    Extend sweep below s={min(scales_done):.2f} or above s={max(scales_done):.2f}")

    # Save results as CSV
    csv_path = sweep_dir / "sweep_results.csv"
    with open(csv_path, "w") as f:
        f.write("scale,xr_final,xr_peak\n")
        for s, traj in results:
            if traj:
                f.write(f"{s},{traj[-1]:.4f},{max(traj):.4f}\n")
            else:
                f.write(f"{s},nan,nan\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
