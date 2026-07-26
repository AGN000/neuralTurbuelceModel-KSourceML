"""Generate 4-panel TBNN coupling summary figure."""
import os; os.chdir('/data/TurbuelceModel')
import sys, numpy as np, torch, h5py
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'openfoam_coupling'); sys.path.insert(0, 'phase2_stress')
from foam_io import read_vector, read_scalar, cell_centres_y, latest_time_dir
from features import phase2_features, NU, DELTA
from tbnn import TBNN
from pathlib import Path

PHASE2 = Path('phase2_stress')
case   = Path('openfoam_coupling/channel_Re550')
tdir   = latest_time_dir(case)
y      = cell_centres_y(100, DELTA, 10.0)

U    = read_vector(tdir / 'U')[:, 0]
k    = read_scalar(tdir / 'k')
om   = read_scalar(tdir / 'omega')
dUdy = np.gradient(U, y)
u_tau  = np.sqrt(NU * abs(dUdy[0]))
Re_tau = u_tau * DELTA / NU
y_plus = y * u_tau / NU
U_plus = U / u_tau

ckpt  = torch.load(str(PHASE2 / 'checkpoints' / 'best_model.pt'), map_location='cpu', weights_only=False)
model = TBNN(**ckpt['config']['model'])
model.load_state_dict(ckpt['model_state'])
model.eval()
stats = ckpt['stats']
mu,  sig = np.array(stats['mu'],     dtype=np.float32), np.array(stats['sigma'],  dtype=np.float32)
lo,  hi  = np.array(stats['lam_lo'], dtype=np.float32), np.array(stats['lam_hi'], dtype=np.float32)

lam_n, T_norm = phase2_features(y, U, k, om, lo, hi)
with torch.no_grad():
    b = model(torch.tensor((lam_n - mu) / sig), torch.tensor(T_norm)).numpy()

b12   = b[:, 1]
R12   = 2.0 * k * b12
floor = 0.01 * np.max(np.abs(dUdy))
denom = np.where(np.abs(dUdy) > floor, dUdy, np.sign(dUdy + 1e-30) * floor)
nut   = np.minimum(np.maximum(-R12 / denom, 0.0), k / np.maximum(om, 1e-10))

with h5py.File('data/channel_Retau550.h5') as f:
    yp_dns   = f['meta/y_plus'][:]
    Up_dns   = f['mean/U_plus'][:]
    nut_dns  = f['closure/nu_t_plus'][:]
    uv_dns   = f['stress/uv'][:]
    k_dns    = f['stress/k'][:]

rmse = np.sqrt(np.mean((np.interp(yp_dns, y_plus, U_plus) - Up_dns)**2))
print(f"Re_tau = {Re_tau:.1f}  (u_tau error {abs(u_tau-8.25e-3)/8.25e-3*100:.1f}%)")
print(f"U+ RMSE = {rmse:.3f}")
print(f"nut+ max = {nut.max()/NU:.2f}  DNS: {nut_dns.max():.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# U+
axes[0,0].semilogx(yp_dns, Up_dns, 'k--', lw=2, label='DNS')
axes[0,0].semilogx(y_plus, U_plus, 'b-',  lw=2, label=f'TBNN  Re_tau={Re_tau:.0f}')
yp_log = np.logspace(np.log10(30), np.log10(600), 60)
axes[0,0].semilogx(yp_log, np.log(yp_log)/0.41 + 5.2, 'r:', lw=1.5, label='Log law')
axes[0,0].set(xlabel='y+', ylabel='U+', title=f'Mean velocity  (U+ RMSE={rmse:.3f})')
axes[0,0].legend(); axes[0,0].grid(True, which='both', alpha=0.3)

# nu_t+
axes[0,1].semilogy(yp_dns, nut_dns,    'k--', lw=2, label='DNS')
axes[0,1].semilogy(y_plus, nut / NU,   'b-',  lw=2, label=f'TBNN (max={nut.max()/NU:.1f})')
axes[0,1].set(xlabel='y+', ylabel='nu_t+', title='Eddy viscosity')
axes[0,1].legend(); axes[0,1].grid(True, which='both', alpha=0.3)

# R12
axes[1,0].plot(yp_dns, uv_dns / u_tau**2, 'k--', lw=2, label='DNS R12')
axes[1,0].plot(y_plus, R12    / u_tau**2, 'b-',  lw=2, label='TBNN R12')
axes[1,0].set(xlabel='y+', ylabel="u'v'/u_tau^2", title='Reynolds shear stress')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# b12
safe_kd  = np.where(k_dns > 1e-30, k_dns, np.nan) * u_tau**2
b12_dns  = np.nan_to_num(uv_dns / (2 * safe_kd))
axes[1,1].plot(yp_dns, b12_dns, 'k--', lw=2, label='DNS b12')
axes[1,1].plot(y_plus, b12,     'b-',  lw=2, label='TBNN b12')
axes[1,1].set(xlabel='y+', ylabel='b12', title='Anisotropy b12')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

fig.suptitle('TBNN Coupling Summary  –  Channel Re_tau=550', fontsize=13, fontweight='bold')
fig.tight_layout()
out = case / 'tbnn_coupling_summary.png'
fig.savefig(str(out), dpi=150)
plt.close(fig)
print(f"Saved: {out}")
