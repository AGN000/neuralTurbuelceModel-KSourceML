"""
Five-way comparison: DNS vs k-omega SST vs TBNN-BFE vs TBNN-BFEk vs NODE-BFE
All cases start from the same converged baseline; BFE/BFEk/NODE have 15 outer iterations.
BFEk adds a k-source correction driving kOmegaSST k toward kpred.
NODE-BFE uses a Neural ODE along streamlines for non-equilibrium b_ij correction.
"""
from __future__ import annotations
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase2_stress"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from foam_io import read_vector, read_scalar, latest_time_dir
from pehill_features import cell_centres_from_polymesh

DNS_FILE  = ROOT / "data" / "pehill_alpha1p0.h5"
SST_CASE  = Path(__file__).resolve().parent / "pehill_alpha1p0"
BFE_CASE  = Path(__file__).resolve().parent / "pehill_alpha1p0_BFE"
BFEK_CASE = Path(__file__).resolve().parent / "pehill_alpha1p0_BFEk"
NODE_CASE = Path(__file__).resolve().parent / "pehill_alpha1p0_NODE"
OUT_DIR   = ROOT / "phase2_stress" / "results_plots"
OUT_DIR.mkdir(exist_ok=True)

STATIONS = [1.0, 2.0, 4.0, 6.0, 7.0, 8.0]  # x/H


def load_dns():
    with h5py.File(DNS_FILE, "r") as f:
        x  = f["meta/x"][:]
        y  = f["meta/y"][:]
        U  = f["mean/U"][:]
        V  = f["mean/V"][:]
    return x, y, U, V


def load_of_case(case_dir: Path):
    cell_xy3, _ = cell_centres_from_polymesh(case_dir)
    cell_xy = cell_xy3[:, :2]
    td = latest_time_dir(case_dir)
    U_vec = read_vector(td / "U")
    U = U_vec[:, 0]
    V = U_vec[:, 1]
    return cell_xy[:, 0], cell_xy[:, 1], U, V


def bulk_velocity(x, y, U, x_probe=0.5, band=0.3):
    """Trapz-integrate U over cross-section near x_probe."""
    mask = np.abs(x - x_probe) < band
    if mask.sum() < 10:
        return float(np.nanmean(U))
    ys = y[mask]; Us = U[mask]
    idx = np.argsort(ys)
    ys, Us = ys[idx], Us[idx]
    H = ys.max() - ys.min()
    return float(np.trapezoid(Us, ys) / H)


def profile_at_x(x, y, U, x_target, band=0.12, ny=50):
    mask = np.abs(x - x_target) < band
    if mask.sum() < 5:
        return None, None
    ys, Us = y[mask], U[mask]
    y_edges = np.linspace(ys.min(), ys.max(), ny + 1)
    y_mid = 0.5 * (y_edges[:-1] + y_edges[1:])
    v_mean = np.array([
        Us[(ys >= y_edges[i]) & (ys < y_edges[i+1])].mean()
        if ((ys >= y_edges[i]) & (ys < y_edges[i+1])).any() else np.nan
        for i in range(ny)
    ])
    return y_mid, v_mean


def reattachment_length(x, y, U, x_start=1.0, x_end=7.5, wall_band=0.35):
    """
    Scan x from x_start to x_end; find FIRST x where near-wall U > 0.
    Near-wall = within wall_band above the local minimum y (handles curved hill wall).
    """
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


def main():
    print("Loading DNS...")
    xd, yd, Ud, Vd = load_dns()
    Ub_dns = bulk_velocity(xd, yd, Ud)
    print(f"  DNS U_bulk = {Ub_dns:.4f}")

    print("Loading k-omega SST...")
    xs, ys, Us, Vs = load_of_case(SST_CASE)
    Ub_sst = bulk_velocity(xs, ys, Us)
    print(f"  SST U_bulk = {Ub_sst:.4f}")

    print("Loading TBNN-BFE...")
    xb, yb, Ub_arr, Vb = load_of_case(BFE_CASE)
    Ub_bfe = bulk_velocity(xb, yb, Ub_arr)
    print(f"  BFE U_bulk = {Ub_bfe:.4f}")

    print("Loading TBNN-BFEk (BFE + k-correction)...")
    xk, yk, Uk_arr, Vk = load_of_case(BFEK_CASE)
    Ub_bfek = bulk_velocity(xk, yk, Uk_arr)
    print(f"  BFEk U_bulk = {Ub_bfek:.4f}")

    print("Loading NODE-BFE (Neural ODE body-force)...")
    xn, yn, Un_arr, Vn = load_of_case(NODE_CASE)
    Ub_node = bulk_velocity(xn, yn, Un_arr)
    print(f"  NODE U_bulk = {Ub_node:.4f}")

    # ── 1. U-profile comparison ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(STATIONS), figsize=(3.5 * len(STATIONS), 5),
                             sharey=True)
    colors = {"DNS": "black",  "k-ω SST": "#1f77b4",  "TBNN-BFE": "#d62728",
              "TBNN-BFEk": "#2ca02c", "NODE-BFE": "#9467bd"}
    lw     = {"DNS": 2.0,     "k-ω SST": 1.5,         "TBNN-BFE": 1.5,
              "TBNN-BFEk": 1.5, "NODE-BFE": 2.0}
    ls     = {"DNS": "-",     "k-ω SST": "--",         "TBNN-BFE": "-",
              "TBNN-BFEk": "-.", "NODE-BFE": ":"}

    datasets = [
        ("DNS",       xd, yd, Ud,     Ub_dns),
        ("k-ω SST",   xs, ys, Us,     Ub_sst),
        ("TBNN-BFE",  xb, yb, Ub_arr, Ub_bfe),
        ("TBNN-BFEk", xk, yk, Uk_arr, Ub_bfek),
        ("NODE-BFE",  xn, yn, Un_arr, Ub_node),
    ]

    for j, x_st in enumerate(STATIONS):
        ax = axes[j]
        for label, x_, y_, U_, Ub_ in datasets:
            yp, Up = profile_at_x(x_, y_, U_, x_st)
            if yp is None:
                continue
            ax.plot(Up / Ub_, yp, color=colors[label], lw=lw[label],
                    ls=ls[label], label=label if j == 0 else None)
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.set_title(f"x/H={x_st:.0f}", fontsize=10)
        ax.set_xlabel("U/U_bulk")
        if j == 0:
            ax.set_ylabel("y/H")
            ax.legend(fontsize=8)

    fig.suptitle("Periodic Hill α=1.0: U/U_bulk profiles", fontsize=12)
    fig.tight_layout()
    out = OUT_DIR / "BFE_comparison_U_profiles.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── 2. Contour comparison ────────────────────────────────────────────────
    xi = np.linspace(0, 9, 300)
    yi = np.linspace(0, 3.035, 120)
    Xi, Yi = np.meshgrid(xi, yi)

    grids = []
    for label, x_, y_, U_, Ub_ in datasets:
        pts = np.column_stack([x_, y_])
        Ui = griddata(pts, U_ / Ub_, (Xi, Yi), method="linear")
        grids.append((label, Ui))

    fig, axes = plt.subplots(5, 1, figsize=(12, 12))
    vmin, vmax = -0.3, 1.2
    for ax, (label, Ui) in zip(axes, grids):
        im = ax.contourf(Xi, Yi, Ui, levels=30, vmin=vmin, vmax=vmax,
                         cmap="RdBu_r")
        cs = ax.contour(Xi, Yi, Ui, levels=[0.0], colors="gold", linewidths=1.5)
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")
        ax.set_ylabel("y/H")
    axes[-1].set_xlabel("x/H")
    fig.colorbar(im, ax=axes, label="U/U_bulk", shrink=0.8)
    fig.suptitle("Periodic Hill α=1.0 — Streamwise Velocity U/U_bulk (gold = U=0 isoline)", fontsize=12)
    out = OUT_DIR / "BFE_comparison_U_contour.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── 3. Reattachment length bar chart ─────────────────────────────────────
    r_dns  = reattachment_length(xd, yd, Ud)
    r_sst  = reattachment_length(xs, ys, Us)
    r_bfe  = reattachment_length(xb, yb, Ub_arr)
    r_bfek = reattachment_length(xk, yk, Uk_arr)
    r_node = reattachment_length(xn, yn, Un_arr)

    print(f"\nReattachment lengths:")
    print(f"  DNS      : {r_dns:.2f} H")
    print(f"  k-ω SST  : {r_sst:.2f} H")
    print(f"  TBNN-BFE : {r_bfe:.2f} H")
    print(f"  TBNN-BFEk: {r_bfek:.2f} H")
    print(f"  NODE-BFE : {r_node:.2f} H")

    fig, ax = plt.subplots(figsize=(8, 4))
    labels_r   = ["DNS", "k-ω SST", "TBNN-BFE", "TBNN-BFEk", "NODE-BFE"]
    vals       = [r_dns, r_sst, r_bfe, r_bfek, r_node]
    bar_colors = ["black", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    bars = ax.bar(labels_r, vals, color=bar_colors, edgecolor="black", width=0.5)
    ax.axhline(r_dns, color="black", ls="--", lw=1.2, label=f"DNS = {r_dns:.2f}H")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}H",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Reattachment length (x/H)")
    ax.set_title("Periodic Hill α=1.0 — Reattachment Length")
    ax.legend()
    ax.set_ylim(0, max(vals) * 1.25)
    fig.tight_layout()
    out = OUT_DIR / "BFE_reattachment_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── 4. Error metric summary ───────────────────────────────────────────────
    print(f"\nError (reattachment) vs DNS:")
    print(f"  k-ω SST  : {abs(r_sst  - r_dns):.2f} H  ({abs(r_sst  - r_dns)/r_dns*100:.1f}%)")
    print(f"  TBNN-BFE : {abs(r_bfe  - r_dns):.2f} H  ({abs(r_bfe  - r_dns)/r_dns*100:.1f}%)")
    print(f"  TBNN-BFEk: {abs(r_bfek - r_dns):.2f} H  ({abs(r_bfek - r_dns)/r_dns*100:.1f}%)")
    print(f"  NODE-BFE : {abs(r_node - r_dns):.2f} H  ({abs(r_node - r_dns)/r_dns*100:.1f}%)")


if __name__ == "__main__":
    main()
