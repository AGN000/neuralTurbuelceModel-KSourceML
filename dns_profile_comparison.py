"""
Compare U, k, Reynolds-stress profiles between
  DNS                      (Xiao para-database, α=1.0)
  Baseline kOmegaSST       (A=0, converged steady state)
  kSourceML A=0.008        (iterative coupling, converged stable FP)

at multiple x-stations: x = 2H, 3H, 4H, 5H, 6H, 7H

Saves:
  - dns_compare_profiles.npz  (raw data per station)
  - dns_compare_panel.png     (plotted profiles)
  - dns_compare_summary.txt   (RMSE/correlation per station)
"""
import json
import numpy as np
import h5py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pehill_features import cell_centres_from_polymesh

ROOT = Path("/data/TurbuelceModel_Loc/openfoam_coupling")
OUT  = Path("/data/TurbuelceModel_Loc/field_inversion_results")
DNS_FILE = "/data/TurbuelceModel_Loc/data/pehill_alpha1p0.h5"

CASE_BASE = ROOT / "pehill_alpha1p0_omegaSrc"
CASE_KSRC = ROOT / "pehill_alpha1p0_kOmegaSSTML"

X_STATIONS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
DX_BIN     = 0.10  # half-width for x-binning


def read_scalar(fp):
    text = fp.read_text()
    if "nonuniform" in text:
        n = text.index("nonuniform")
        op = text.index("(\n", n) + 2
        cl = text.index("\n)\n", op)
        return np.fromstring(text[op:cl], sep="\n")
    elif "uniform" in text:
        v = float(text.split("uniform")[1].split(";")[0])
        return np.full(1, v)


def read_vector(fp):
    text = fp.read_text()
    if "nonuniform" in text:
        n = text.index("nonuniform")
        op = text.index("(\n", n) + 2
        cl = text.index("\n)\n", op)
        return np.fromstring(text[op:cl].replace("(", "").replace(")", ""),
                             sep="\n").reshape(-1, 3)
    return None


def latest_time_dir(case):
    return max((d for d in case.iterdir()
                if d.is_dir() and d.name.replace('.', '', 1).isdigit()
                and d.name != "0_warm"),
               key=lambda d: float(d.name))


def get_profile(cell_xy, field, x0, dx=DX_BIN):
    x = cell_xy[:, 0]; y = cell_xy[:, 1]
    m = np.abs(x - x0) <= dx
    if m.sum() < 5:
        return np.array([]), np.array([])
    order = np.argsort(y[m])
    return y[m][order], field[m][order]


# ─── Load mesh and DNS ────────────────────────────────────────────────────────
cc, _ = cell_centres_from_polymesh(CASE_BASE)
cell_xy = cc[:, :2].astype(np.float64)
N = len(cell_xy)

with h5py.File(DNS_FILE, 'r') as f:
    dns = {
        "U": f['mean/U'][:], "V": f['mean/V'][:],
        "k": f['stress/k'][:], "R12": f['stress/R12'][:],
        "x": f['meta/x'][:], "y": f['meta/y'][:],
    }

# ─── Load RANS fields ────────────────────────────────────────────────────────
print(f"Reading baseline (A=0) from {CASE_BASE}/0_warm")
U_base = read_vector(CASE_BASE / "0_warm" / "U")
k_base = read_scalar(CASE_BASE / "0_warm" / "k")

print(f"Reading kSourceML (A=0.008, iterative-stable) from latest_time_dir")
latest = latest_time_dir(CASE_KSRC)
print(f"  latest: {latest.name}")
U_ksrc = read_vector(latest / "U")
k_ksrc = read_scalar(latest / "k")

if U_base.shape[0] == 1: U_base = np.tile(U_base, (N, 1))
if U_ksrc.shape[0] == 1: U_ksrc = np.tile(U_ksrc, (N, 1))

print(f"\n{'x_station':>10}  {'metric':>14}  {'baseline':>10}  {'kSourceML':>10}  {'DNS':>10}  {'base_err':>9}  {'ksrc_err':>9}")
print("-" * 100)

results_per_x = {}
panel_data = {"x_stations": X_STATIONS}

for x0 in X_STATIONS:
    # Use DNS x,y for this station — DNS data uses the SAME mesh as RANS
    y_dns, U_dns_p = get_profile(cell_xy, dns["U"], x0)
    _,    k_dns_p  = get_profile(cell_xy, dns["k"], x0)
    _,    Ub_p     = get_profile(cell_xy, U_base[:, 0], x0)
    _,    kb_p     = get_profile(cell_xy, k_base,        x0)
    _,    Uk_p     = get_profile(cell_xy, U_ksrc[:, 0], x0)
    _,    kk_p     = get_profile(cell_xy, k_ksrc,        x0)

    if len(y_dns) < 5: continue

    # Compute RMSE in U and k (normalised)
    U_scale = max(abs(U_dns_p).max(), 1e-9)
    rmse_U_base = float(np.sqrt(np.mean((Ub_p - U_dns_p) ** 2)) / U_scale)
    rmse_U_ksrc = float(np.sqrt(np.mean((Uk_p - U_dns_p) ** 2)) / U_scale)
    k_scale = max(abs(k_dns_p).max(), 1e-12)
    rmse_k_base = float(np.sqrt(np.mean((kb_p - k_dns_p) ** 2)) / k_scale)
    rmse_k_ksrc = float(np.sqrt(np.mean((kk_p - k_dns_p) ** 2)) / k_scale)

    results_per_x[x0] = {
        "U_rmse_baseline": rmse_U_base, "U_rmse_kSourceML": rmse_U_ksrc,
        "k_rmse_baseline": rmse_k_base, "k_rmse_kSourceML": rmse_k_ksrc,
        "n_points": int(len(y_dns)),
    }

    panel_data[f"x{x0}"] = {
        "y": y_dns.tolist(),
        "U_dns":  U_dns_p.tolist(),  "U_base": Ub_p.tolist(), "U_ksrc": Uk_p.tolist(),
        "k_dns":  k_dns_p.tolist(),  "k_base": kb_p.tolist(), "k_ksrc": kk_p.tolist(),
    }

    print(f"{x0:>10.2f}  {'U RMSE/U_max':>14}  "
          f"{rmse_U_base:>10.3f}  {rmse_U_ksrc:>10.3f}  {0.0:>10.3f}  "
          f"{rmse_U_base:>9.3f}  {rmse_U_ksrc:>9.3f}")
    print(f"{'':>10}  {'k RMSE/k_max':>14}  "
          f"{rmse_k_base:>10.3f}  {rmse_k_ksrc:>10.3f}  {0.0:>10.3f}  "
          f"{rmse_k_base:>9.3f}  {rmse_k_ksrc:>9.3f}")

# Aggregate
all_U_base = np.mean([r["U_rmse_baseline"] for r in results_per_x.values()])
all_U_ksrc = np.mean([r["U_rmse_kSourceML"] for r in results_per_x.values()])
all_k_base = np.mean([r["k_rmse_baseline"] for r in results_per_x.values()])
all_k_ksrc = np.mean([r["k_rmse_kSourceML"] for r in results_per_x.values()])

print("\n" + "=" * 80)
print("AGGREGATE (mean RMSE across all stations):")
print(f"  U/U_max:  baseline={all_U_base:.3f}  kSourceML={all_U_ksrc:.3f}  "
      f"improvement={100*(1-all_U_ksrc/max(all_U_base,1e-9)):.1f}%")
print(f"  k/k_max:  baseline={all_k_base:.3f}  kSourceML={all_k_ksrc:.3f}  "
      f"improvement={100*(1-all_k_ksrc/max(all_k_base,1e-9)):.1f}%")

# Save
np.savez(OUT / "dns_compare_profiles.npz", **{
    f"x{x0}_y": np.array(panel_data[f"x{x0}"]["y"])
    for x0 in X_STATIONS if f"x{x0}" in panel_data
}, **{
    f"x{x0}_{key}": np.array(panel_data[f"x{x0}"][key])
    for x0 in X_STATIONS if f"x{x0}" in panel_data
    for key in ["U_dns", "U_base", "U_ksrc", "k_dns", "k_base", "k_ksrc"]
})

with open(OUT / "dns_compare_summary.json", "w") as f:
    json.dump({
        "per_station": {f"x={x0}": v for x0, v in results_per_x.items()},
        "aggregate": {
            "U_rmse_baseline": float(all_U_base),
            "U_rmse_kSourceML": float(all_U_ksrc),
            "U_improvement_pct": float(100 * (1 - all_U_ksrc / max(all_U_base, 1e-9))),
            "k_rmse_baseline": float(all_k_base),
            "k_rmse_kSourceML": float(all_k_ksrc),
            "k_improvement_pct": float(100 * (1 - all_k_ksrc / max(all_k_base, 1e-9))),
        }
    }, f, indent=2)

print(f"\nSaved → {OUT}/dns_compare_profiles.npz")
print(f"Saved → {OUT}/dns_compare_summary.json")

# ─── Plot ────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(X_STATIONS), figsize=(3 * len(X_STATIONS), 7),
                              sharey=True)
    for i, x0 in enumerate(X_STATIONS):
        if f"x{x0}" not in panel_data: continue
        d = panel_data[f"x{x0}"]
        y = np.array(d["y"])
        # U row
        axes[0, i].plot(np.array(d["U_dns"]),  y, 'k-', lw=2, label='DNS')
        axes[0, i].plot(np.array(d["U_base"]), y, 'b--', label='kOmegaSST')
        axes[0, i].plot(np.array(d["U_ksrc"]), y, 'r-', label='kSourceML')
        axes[0, i].set_title(f"x = {x0:.1f}H")
        axes[0, i].axvline(0, color='gray', lw=0.5)
        axes[0, i].set_xlabel("U")
        if i == 0:
            axes[0, i].set_ylabel("y/H")
            axes[0, i].legend(fontsize=8)
        # k row
        axes[1, i].plot(np.array(d["k_dns"]),  y, 'k-', lw=2)
        axes[1, i].plot(np.array(d["k_base"]), y, 'b--')
        axes[1, i].plot(np.array(d["k_ksrc"]), y, 'r-')
        axes[1, i].set_xlabel("k")
        if i == 0:
            axes[1, i].set_ylabel("y/H")
    plt.suptitle("DNS vs baseline kOmegaSST vs kSourceML A=0.008 — pehill α=1.0", y=1.02)
    plt.tight_layout()
    out_png = OUT / "dns_compare_panel.png"
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f"Saved → {out_png}")
except ImportError as e:
    print(f"Plot skipped (matplotlib): {e}")
