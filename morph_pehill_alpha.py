"""
Generate α-specific pehill meshes by x-morphing the existing α=1.0 mesh.

Approach:
- Reads pehill_alpha1p0_omegaSrc/constant/polyMesh/points
- Applies a piecewise-linear x-mapping that stretches/compresses the hill
  regions while keeping the domain length fixed at L_x = 9 H:
    Old left hill  [0,         x_h0]      → new [0,         α·x_h0]
    Old flat       [x_h0,      L_x - x_h0] → new [α·x_h0,   L_x - α·x_h0]
    Old right hill [L_x - x_h0, L_x]      → new [L_x - α·x_h0, L_x]
- y, z unchanged. Topology (faces/owner/neighbour/boundary) unchanged.

Usage:
    python morph_pehill_alpha.py --alpha 0.5 --out pehill_alpha0p5_omegaSrc
"""
import argparse
import shutil
from pathlib import Path
import numpy as np
import re

SOURCE = Path("/data/TurbuelceModel_Loc/openfoam_coupling/pehill_alpha1p0_omegaSrc")
ROOT   = Path("/data/TurbuelceModel_Loc/openfoam_coupling")
LX     = 9.0
X_HILL_END = 1.832   # End of left hill in α=1.0 baseline
HILL_HEIGHT = 1.0
DOMAIN_HEIGHT = 3.034


def read_points(fpath):
    text = fpath.read_text()
    n = text.index("(\n") + 2
    end = text.index("\n)\n", n)
    pts = np.fromstring(text[n:end].replace("(", " ").replace(")", " "),
                        sep=" ").reshape(-1, 3)
    return pts, text[:n], text[end:]


def write_points(fpath, pts, header, tail):
    body = "\n".join(f"({p[0]} {p[1]} {p[2]})" for p in pts)
    fpath.write_text(header + body + tail)


def x_morph(x_old, alpha):
    """Piecewise-linear x morphing for α scaling."""
    x_h0_old = X_HILL_END
    x_h1_old = LX - X_HILL_END
    x_h0_new = alpha * X_HILL_END
    x_h1_new = LX - alpha * X_HILL_END

    x_new = np.zeros_like(x_old)

    # Left hill: [0, x_h0_old] → [0, x_h0_new]
    left = x_old <= x_h0_old
    x_new[left] = x_old[left] * (x_h0_new / x_h0_old)

    # Right hill: [x_h1_old, LX] → [x_h1_new, LX]
    right = x_old >= x_h1_old
    x_new[right] = x_h1_new + (x_old[right] - x_h1_old) * \
                   ((LX - x_h1_new) / (LX - x_h1_old))

    # Flat: [x_h0_old, x_h1_old] → [x_h0_new, x_h1_new]
    mid = ~left & ~right
    x_new[mid] = x_h0_new + (x_old[mid] - x_h0_old) * \
                 ((x_h1_new - x_h0_new) / (x_h1_old - x_h0_old))

    return x_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True, help="case directory name")
    args = ap.parse_args()

    src = SOURCE
    dst = ROOT / args.out
    if dst.exists():
        print(f"Removing existing {dst}")
        shutil.rmtree(dst)
    print(f"Cloning {src} → {dst}")
    shutil.copytree(src, dst)

    # Clean numeric time dirs except 0 and 0_warm (we'll regenerate fields too)
    for d in dst.iterdir():
        if (d.is_dir() and d.name.replace('.', '', 1).isdigit()
                and d.name != "0_warm"):
            shutil.rmtree(d)
    # Keep 0/ as a copy of 0_warm
    if not (dst / "0").exists():
        shutil.copytree(dst / "0_warm", dst / "0")

    # Read points
    pts_file = dst / "constant" / "polyMesh" / "points"
    pts, header, tail = read_points(pts_file)
    print(f"Read {len(pts)} mesh points")
    x_old = pts[:, 0].copy()

    x_new = x_morph(x_old, args.alpha)
    pts[:, 0] = x_new
    write_points(pts_file, pts, header, tail)
    print(f"Wrote morphed points (α={args.alpha})")
    print(f"  Old hill end: {X_HILL_END:.3f}H  → new: {args.alpha*X_HILL_END:.3f}H")
    print(f"  Domain x range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]H")

    print(f"\nCase ready: {dst}")
    print(f"Run baseline RANS with: simpleFoam (after blockMesh validation)")


if __name__ == "__main__":
    main()
