"""
Closed-loop iterative kSourceML coupling.

At each outer iteration:
  1. Read k from the latest OF time directory
   2. kSourceML = A * k  
  3. Run simpleFoam for INNER iterations
4. Report x_r

Verifies that A=0.008 is a stable fixed point giving x_r≈4.10H.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pehill_features import cell_centres_from_polymesh

CASE    = Path("pehill_case")
OF_ROOT = "/opt/openfoam11"
OUT_DIR = Path("results")

SL_XMIN, SL_XMAX = 1.0, 5.0
SL_YMIN, SL_YMAX = 0.1, 1.0


def of_header(class_, loc, name):
    return (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  =========                 |\n"
        "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox\n"
        "   \\\\    /   O peration     | Version:  11\n"
        "\\*---------------------------------------------------------------------------*/\n"
        f"FoamFile\n{{\n    format      ascii;\n    class       {class_};\n"
        f"    location    \"{loc}\";\n    object      {name};\n}}\n"
        "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
    )


def write_scalar_field(fpath, data, dims, name, time_name):
    lines = [of_header("volScalarField", time_name, name),
             f"\ndimensions      {dims};\n",
             f"\ninternalField   nonuniform List<scalar>\n{len(data)}\n(\n"]
    lines.extend(f"{v:.8e}\n" for v in data)
    lines.append(")\n;\n\nboundaryField\n{\n"
                 "    bottomWall { type zeroGradient; }\n"
                 "    topWall    { type zeroGradient; }\n"
                 "    inlet      { type cyclic; }\n"
                 "    outlet     { type cyclic; }\n"
                 "    defaultFaces { type empty; }\n"
                 "}\n\n// *** //\n")
    tmp = fpath.with_suffix(".tmp")
    tmp.write_text("".join(lines))
    tmp.rename(fpath)


def read_scalar(fpath, N):
    try:
        text = fpath.read_text()
    except OSError:
        return None
    if "nonuniform" in text:
        nu = text.index("nonuniform")
        op = text.index("(\n", nu) + 2
        cl = text.index("\n)\n", op)
        return np.fromstring(text[op:cl], sep="\n")
    elif "uniform" in text:
        val = float(text.split("uniform")[1].split(";")[0])
        return np.full(N, val)
    return None


def read_vector(fpath, N):
    try:
        text = fpath.read_text()
    except OSError:
        return None
    if "nonuniform" in text:
        nu = text.index("nonuniform")
        op = text.index("(\n", nu) + 2
        cl = text.index("\n)\n", op)
        blk = text[op:cl].replace("(", "").replace(")", "")
        return np.fromstring(blk, sep="\n").reshape(-1, 3)
    elif "uniform" in text:
        inner = text.split("uniform")[1].split("(")[1].split(")")[0]
        return np.array([[float(v) for v in inner.split()]])
    return None


def set_end_time(case, end_time):
    cd = case / "system" / "controlDict"
    text = cd.read_text()
    text = re.sub(r'endTime\s+\d+\s*;',      f'endTime         {end_time};', text)
    text = re.sub(r'writeInterval\s+\d+\s*;', f'writeInterval   {end_time};', text)
    text = re.sub(r'startFrom\s+\w+\s*;', 'startFrom       latestTime;', text)
    cd.write_text(text)


def latest_time_dir(case):
    dirs = [d for d in case.iterdir()
            if d.is_dir() and d.name.replace('.', '', 1).isdigit()]
    return max(dirs, key=lambda d: float(d.name))


def reattachment_length(cell_xy, U, x_start=1.0, x_end=7.5, wall_band=0.35):
    x = cell_xy[:, 0]; y = cell_xy[:, 1]
    dx = 0.05
    for xb in np.arange(x_start, x_end + dx, dx):
        mask = np.abs(x - xb) < dx
        if mask.sum() < 2: continue
        y_wall = y[mask].min()
        if y_wall > 0.05: continue
        near = mask & (y < y_wall + wall_band)
        if near.sum() < 2: continue
        if U[near].mean() > 0:
            return float(xb)
    return float(x_end)


def check_convergence(log_path):
    try:
        lines = log_path.read_text().splitlines()
        for line in reversed(lines):
            if "Solving for Ux" in line and "Initial residual" in line:
                m = re.search(r'Initial residual = ([0-9.eE+\-]+)', line)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case",    type=Path, default=CASE,   help="OpenFOAM case directory")
    p.add_argument("--of_env",  type=str,  default=OF_ROOT, help="OpenFOAM installation root")
    p.add_argument("--out_dir", type=Path, default=OUT_DIR, help="Output directory for logs & results")
    p.add_argument("--A",       type=float, default=0.008,  help="Source amplitude [1/s]")
    p.add_argument("--outer",   type=int,   default=10,     help="Outer coupling iterations")
    p.add_argument("--inner",   type=int,   default=3000,   help="SimpleFoam iters per outer")
    p.add_argument("--reset",   action="store_true",        help="Reset to warm-start before starting")
    args = p.parse_args()

    case    = args.case.resolve()
    of_root = args.of_env
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "iterate_kSourceML.log"
    log_path.write_text("")

    # Load mesh
    cell_xy3, _ = cell_centres_from_polymesh(case)
    cell_xy = cell_xy3[:, :2].astype(np.float64)
    N = len(cell_xy)
    x = cell_xy[:, 0]; y = cell_xy[:, 1]
    shear_mask = ((x >= SL_XMIN) & (x <= SL_XMAX) &
                  (y >= SL_YMIN) & (y <= SL_YMAX))
    print(f"Mesh: {N} cells  shear-layer: {shear_mask.sum()} cells")
    print(f"A={args.A:.4f} 1/s  outer={args.outer}  inner={args.inner}")

    # Optionally reset to warm start
    if args.reset:
        print("Resetting to warm start...")
        for d in case.iterdir():
            if d.is_dir() and d.name.replace('.', '', 1).isdigit() and d.name != "0":
                shutil.rmtree(d)
        for src in (case / "0_warm").iterdir():
            shutil.copy2(src, case / "0" / src.name)

    current_t = int(float(latest_time_dir(case).name))
    print(f"Starting from t={current_t}")

    results = []
    print(f"\n{'outer':>6}  {'t':>8}  {'x_r':>8}  {'k_mean':>10}  {'src_mean':>12}  {'res':>10}")
    print("-" * 65)

    for outer in range(1, args.outer + 1):
        latest = latest_time_dir(case)
        k_rans = read_scalar(latest / "k", N)
        if k_rans is None or len(k_rans) != N:
            print(f"  ERROR: cannot read k at t={latest.name}"); break

        # Compute and write kSourceML
        k_src = np.where(shear_mask, args.A * k_rans, 0.0)
        zeros = np.zeros(N)
        time_name = latest.name
        write_scalar_field(latest / "kSourceML",    k_src, "[0 2 -3 0 0 0 0]",
                           "kSourceML", time_name)
        write_scalar_field(latest / "omegaSourceML", zeros, "[0 0 -2 0 0 0 0]",
                           "omegaSourceML", time_name)

        # Run simpleFoam
        set_end_time(case, current_t + args.inner)
        cmd = (f"source {of_root}/etc/bashrc 2>/dev/null; "
               f"cd {case} && simpleFoam >> {log_path} 2>&1; exit $?")
        rc = subprocess.run(["bash", "-c", cmd]).returncode
        current_t += args.inner

        if rc != 0:
            print(f"  outer={outer}: simpleFoam failed (rc={rc})"); break

        # Read result
        latest = latest_time_dir(case)
        U_vec = read_vector(latest / "U", N)
        if U_vec is None: break
        if U_vec.shape[0] == 1: U_vec = np.tile(U_vec, (N, 1))
        xr = reattachment_length(cell_xy, U_vec[:, 0])

        k_new = read_scalar(latest / "k", N)
        k_mean = k_new.mean() if k_new is not None else 0.0
        src_mean = k_src[shear_mask].mean()
        res = check_convergence(log_path)

        print(f"{outer:>6}  {current_t:>8}  {xr:>7.3f}H  {k_mean:>10.4e}  {src_mean:>12.4e}  {res if res else 'N/A':>10}")
        results.append({"outer": outer, "t": current_t, "xr": xr,
                        "k_mean": float(k_mean), "src_mean": float(src_mean),
                        "res": float(res) if res else None, "rc": rc})

    # Summary
    if results:
        xrs = [r["xr"] for r in results]
        print(f"\nSummary: x_r range=[{min(xrs):.3f}, {max(xrs):.3f}]H  "
              f"final={xrs[-1]:.3f}H  (target=4.10H)")
        with open(out_dir / "iterate_kSourceML_results.json", "w") as f:
            json.dump({"A": args.A, "results": results}, f, indent=2)
        print(f"Results → {out_dir}/iterate_kSourceML_results.json")


if __name__ == "__main__":
    main()
