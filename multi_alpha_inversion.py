"""
Multi-α kSourceML field inversion: run kSourceML with several A values
on each of {0.5, 0.8, 1.0, 1.2, 1.5}-α meshes, find optimal A*(α).

Each (α, A) eval = one simpleFoam run. Total: 5 α × 4 A = 20 runs.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pehill_features import cell_centres_from_polymesh

OF_ROOT = "/opt/openfoam11"
INNER   = 6000
A_VALUES = [0.000, 0.005, 0.008, 0.012, 0.016]
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


def write_scalar_field(fp, data, dims, name, time_name):
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
    tmp = fp.with_suffix(".tmp")
    tmp.write_text("".join(lines)); tmp.rename(fp)


def read_scalar(fp, N):
    text = fp.read_text()
    if "nonuniform" in text:
        nu = text.index("nonuniform")
        op = text.index("(\n", nu) + 2
        cl = text.index("\n)\n", op)
        return np.fromstring(text[op:cl], sep="\n")
    elif "uniform" in text:
        return np.full(N, float(text.split("uniform")[1].split(";")[0]))
    return None


def read_vector(fp):
    text = fp.read_text()
    if "nonuniform" in text:
        nu = text.index("nonuniform")
        op = text.index("(\n", nu) + 2
        cl = text.index("\n)\n", op)
        blk = text[op:cl].replace("(", "").replace(")", "")
        return np.fromstring(blk, sep="\n").reshape(-1, 3)
    return None


def latest_time_dir(case):
    dirs = [d for d in case.iterdir()
            if d.is_dir() and d.name.replace('.', '', 1).isdigit()
            and d.name != "0_warm"]
    return max(dirs, key=lambda d: float(d.name))


def reattachment_length(cell_xy, U, x_start, x_end, wall_band=0.35):
    x = cell_xy[:, 0]; y = cell_xy[:, 1]
    dx = 0.05
    for xb in np.arange(x_start, x_end + dx, dx):
        mask = np.abs(x - xb) < dx
        if mask.sum() < 2: continue
        y_wall = y[mask].min()
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


def set_end_time(case, end_time):
    cd = case / "system" / "controlDict"
    text = cd.read_text()
    text = re.sub(r'endTime\s+\d+\s*;', f'endTime         {end_time};', text)
    text = re.sub(r'writeInterval\s+\d+\s*;', f'writeInterval   {end_time};', text)
    text = re.sub(r'startFrom\s+\w+\s*;', 'startFrom       startTime;', text)
    text = re.sub(r'startTime\s+\d+\s*;', 'startTime       0;', text)
    cd.write_text(text)


def run_eval(case, alpha, A, eval_name):
    log_dir = OUT_DIR / "multi_alpha_logs"
    log_dir.mkdir(exist_ok=True)
    log = log_dir / f"{eval_name}.log"

    cell_xy3, _ = cell_centres_from_polymesh(case)
    cell_xy = cell_xy3[:, :2].astype(np.float64)
    N = len(cell_xy)
    x = cell_xy[:, 0]; y = cell_xy[:, 1]

    # Shear layer: x in [hill_end, hill_start_right], y in [0.1, 1.0]
    x_hill_end = alpha * 1.832
    x_hill_start_r = 9.0 - alpha * 1.832
    sl_mask = ((x >= x_hill_end) & (x <= x_hill_start_r)
               & (y >= SL_YMIN) & (y <= SL_YMAX))

    # Clean and refresh
    for d in case.iterdir():
        if (d.is_dir() and d.name.replace('.', '', 1).isdigit()
                and d.name not in ("0", "0_warm")):
            shutil.rmtree(d)
    for src in (case / "0_warm").iterdir():
        if src.is_file():
            shutil.copy2(src, case / "0" / src.name)

    k_baseline = read_scalar(case / "0_warm" / "k", N)

    k_src = np.where(sl_mask, A * k_baseline, 0.0)
    write_scalar_field(case / "0" / "kSourceML",
                       k_src, "[0 2 -3 0 0 0 0]", "kSourceML", "0")
    write_scalar_field(case / "0" / "omegaSourceML",
                       np.zeros(N), "[0 0 -2 0 0 0 0]", "omegaSourceML", "0")

    set_end_time(case, INNER)
    cmd = (f"source {OF_ROOT}/etc/bashrc 2>/dev/null; export FOAM_MPI=mpich-3.3; "
           f"cd {case} && simpleFoam >> {log} 2>&1; exit $?")
    rc = subprocess.run(["bash", "-c", cmd]).returncode
    if rc != 0:
        return 7.5, False, rc, log, sl_mask.sum()

    latest = latest_time_dir(case)
    U_vec = read_vector(latest / "U")
    if U_vec is None:
        return 7.5, False, rc, log, sl_mask.sum()
    if U_vec.shape[0] == 1:
        U_vec = np.tile(U_vec, (N, 1))

    # Reattachment search starts after left hill ends
    xr = reattachment_length(cell_xy, U_vec[:, 0],
                              x_start=x_hill_end + 0.1, x_end=x_hill_start_r)
    final_res = check_convergence(log)
    early_stop = float(latest.name) < INNER
    converged = early_stop or (final_res is not None and final_res < 1e-4)
    return xr, converged, rc, log, sl_mask.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--of_env", default="/opt/openfoam11", help="OpenFOAM installation root")
    parser.add_argument("--cases_dir", default=".", type=Path, help="Directory containing pehill_alpha* case dirs")
    parser.add_argument("--out_dir", default="results", type=Path, help="Output directory")
    args = parser.parse_args()

    global OF_ROOT
    if args.of_env:
        OF_ROOT = args.of_env
    root = args.cases_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ALPHAS = [
        (0.5, root / "pehill_alpha0p5_kSourceML"),
        (0.8, root / "pehill_alpha0p8_kSourceML"),
        (1.0, root / "pehill_alpha1p0_kSourceML"),
        (1.2, root / "pehill_alpha1p2_kSourceML"),
        (1.5, root / "pehill_alpha1p5_kSourceML"),
    ]
    A_VALUES = [0.000, 0.005, 0.008, 0.012, 0.016]

    print("=" * 80)
    print("Multi-α kSourceML field inversion")
    print("=" * 80)
    print(f"{'α':>5}  {'case':<35}  {'A':>8}  {'x_r':>7}  {'sl':>5}  {'res':>9}")
    print("-" * 80)

    results = []
    for alpha, case in ALPHAS:
        for A in A_VALUES:
            xr, conv, rc, log, sl_n = run_eval(case, alpha, A,
                                                 f"a{alpha:.2f}_A{A:.4f}")
            res = check_convergence(log)
            status = "ok" if conv else (f"rc={rc}" if rc else f"r={res:.1e}")
            results.append({"alpha": alpha, "case": str(case),
                            "A": A, "xr": xr, "sl_cells": int(sl_n),
                            "converged": conv, "rc": rc, "res": res})
            print(f"{alpha:5.2f}  {case.name:<35}  {A:8.4f}  {xr:6.2f}H  "
                  f"{sl_n:5d}  {res if res else 999:.1e}")

    with open(out_dir / "multi_alpha_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_dir}/multi_alpha_results.json")

    # Per-α best A
    print("\n=== Best A* per α (closest x_r to 4.10H) ===")
    by_alpha = {}
    for r in results:
        if r["converged"]:
            by_alpha.setdefault(r["alpha"], []).append(r)
    for alpha, rs in sorted(by_alpha.items()):
        best = min(rs, key=lambda r: abs(r["xr"] - 4.10))
        print(f"  α={alpha}:  A*={best['A']:.4f}  x_r={best['xr']:.2f}H")


if __name__ == "__main__":
    main()
