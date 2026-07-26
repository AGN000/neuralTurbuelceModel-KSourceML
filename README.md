# KSourceML 

Proportional k-source injection inside kOmegaSST for RANS turbulence closure correction.
**The only method in this project achieving a stable fixed point near the DNS target.**

| Method | x_r/H | Error | Stability |
|--------|-------|-------|-----------|
| Standard kOmegaSST | 6.20–6.25 | +52% | Stable (wrong FP) |
| **kSourceML A=0.008 (iterative)** | **4.00** | **2.4%** | **Stable FP** |
| **kSourceML A=0.008 (one-shot FI)** | **4.10** | **0.0%** | **Stable FP** |
| ML predictor (RF binary mask) | 4.00 | 2.4% | Stable FP |

---

## Prerequisites

- **OpenFOAM 11** — install from https://openfoam.org or use the NLSS fork:
  <https://gitlab.ethz.ch/nlss/nlss-openfoam>
- **Python 3.10+** with:
  ```bash
  pip install numpy scipy h5py pyyaml scikit-learn matplotlib
  ```
- **DNS dataset** — periodic hill data (Wu/Xiao 2020):
  ```bash
  python download_pehill.py --cases 1p0
  ```
  Or download from <https://github.com/xiaoh/para-database-for-PIML>

---

## Step-by-step workflow

### Step 1: Build the kOmegaSSTML solver

```bash
source /opt/openfoam11/etc/bashrc
cd kOmegaSSTML
wmake libso
```

This compiles `libkOmegaSSTML.so` into `$FOAM_USER_LIBBIN`.

### Step 2: Set up the OpenFOAM case

```bash
# Copy the case template
cp -r case_template my_run

# Copy or link the DNS HDF5 data
cp /path/to/pehill_alpha1p0.h5 my_run/
```

The template contains:
- `constant/polyMesh/` — 14751-cell periodic hill mesh (α=1.0)
- `constant/momentumTransport` — configured for `kOmegaSSTML`
- `0/` — initial conditions (k, omega, p, U, nut)
- `0_warm/` — warm-start fields (used with `--reset`)

### Step 3: Run the iterative kSourceML coupling

```bash
source /opt/openfoam11/etc/bashrc
python3 iterate_kSourceML.py \
    --case my_run \
    --A 0.008 \
    --outer 10 \
    --inner 3000
```

**What happens per outer iteration:**
1. Reads current k field from the latest time directory
2. Computes `kSourceML = A * k_RANS` in the shear layer (x∈[1,5]H, y∈[0.1,1.0]H)
3. Writes `kSourceML` as a `volScalarField` into the current time directory
4. Runs `simpleFoam` for 3000 steps
5. Reports reattachment length x_r

**Expected output:**
```
outer      t      x_r/H      k_mean       src_mean        res
     1     3000    4.55H    2.5015e-06    2.0012e-08    ...
     2     6000    4.10H    9.8918e-06    7.9134e-08    ...
     3     9000    4.00H    1.0964e-05    8.7712e-08    ...
     4    12000    4.00H    1.1076e-05    8.8608e-08    ...
```

### Step 4: Cross-slope testing (α=0.5–1.5)

Generate morphed meshes:
```bash
python3 morph_pehill_alpha.py --alpha 0.5 --out pehill_alpha0p5
python3 morph_pehill_alpha.py --alpha 0.8 --out pehill_alpha0p8
python3 morph_pehill_alpha.py --alpha 1.2 --out pehill_alpha1p2
python3 morph_pehill_alpha.py --alpha 1.5 --out pehill_alpha1p5
```

Then run cross-slope field inversion:
```bash
python3 multi_alpha_inversion.py \
    --cases_dir . \
    --of_env /opt/openfoam11
```

### Step 5: Train an ML predictor (optional)

Train a Random Forest to predict the A-field from RANS features:
```bash
python3 train_kSourceML_predictor.py \
    --data_dir results \
    --out_dir results
```

---

## Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Source amplitude A | 0.008 s⁻¹ | For α=1.0 (also works for α=1.2) |
| A for α=1.5 | 0.012 s⁻¹ | 50% increase for steeper hill |
| Shear layer | x∈[1,5]H, y∈[0.1,1.0]H | Where k-source is applied |
| Inner iterations | 3000 | simpleFoam steps per outer loop |
| Relaxation | None (direct) | kSourceML is an implicit source in kEqn |

---

## File reference

| File | Purpose |
|------|---------|
| `kOmegaSSTML/` | C++ OpenFOAM solver (wmake libso) |
| `iterate_kSourceML.py` | **Main script** — proportional A·k coupling |
| `couple_kOmegaSSTML_pehill.py` | Alternative: magnitude-matching λ·(k_DNS − k_RANS) |
| `pehill_features.py` | Cell-centre computation from polyMesh (required import) |
| `multi_alpha_inversion.py` | Cross-slope A*(α) field inversion |
| `morph_pehill_alpha.py` | Generate α-specific meshes |
| `train_kSourceML_predictor.py` | Train RF/MLP predictor for A-field |
| `field_inversion_sweep.py` | Rij scaling sweep (archived) |
| `download_pehill.py` | DNS data download from GitHub |
| `case_template/` | Clean periodic hill case (no output time dirs) |

---

## Acknowledgments

- **NLSS OpenFOAM** — <https://gitlab.ethz.ch/nlss/nlss-openfoam>
- **Periodic hill DNS dataset** — Xiao, H., Wu, J.-L., Laizet, S., & Duan, L. (2020).
  *Flows over periodic hills of parameterized geometries: A dataset for data-driven
  turbulence modeling from direct simulations.* **Computers & Fluids**, 200, 104431.
  DOI: <https://doi.org/10.1016/j.compfluid.2020.104431>
  Dataset: <https://github.com/xiaoh/para-database-for-PIML>
- **Standard periodic hill DNS reference** — Breuer, M., Peller, N., Rapp, C., &
  Manhart, M. (2009). *Flow over periodic hills — numerical and experimental study
  in a wide range of Reynolds numbers.* **Computers & Fluids**, 38, 433–457.
