import json
import numpy as np
from pathlib import Path
import sys
import pickle

OUT_DIR = Path("results")

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--data_dir", type=Path, default=OUT_DIR)
_parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
_args = _parser.parse_args()

OUT_DIR = _args.out_dir
DATA_DIR = _args.data_dir

data = np.load(DATA_DIR / "fi_analysis.npz")
cell_xy     = data["cell_xy"]        # (N, 2)
k_warm      = data["k_warm"]         # (N,)
omega_warm  = data["omega_warm"]     # (N,)
nut_warm    = data["nut_warm"]       # (N,)
shear_mask  = data["shear_mask"]     # (N,) bool
A_field     = data["A_field"]        # (N,) 0 or 0.008
best_A      = float(data["best_A"].flat[0])

N = len(cell_xy)
x = cell_xy[:, 0]; y = cell_xy[:, 1]
NU = 7.0e-6   # kinematic viscosity

print(f"Loaded {N} cells  shear-layer: {shear_mask.sum()} cells")
print(f"best_A = {best_A:.4f} 1/s")

eps_safe = 1e-30

features_dict = {
    "x":            x,
    "y":            y,
    "k":            k_warm,
    "omega":        omega_warm,
    "nut":          nut_warm,
    "k_over_omega": k_warm / np.maximum(omega_warm, eps_safe),       # ~ eddy length²
    "nut_over_nu":  nut_warm / NU,                                    # turbulent Re
    "log_k":        np.log(np.maximum(k_warm, eps_safe)),
    "log_omega":    np.log(np.maximum(omega_warm, eps_safe)),
    "log_nut":      np.log(np.maximum(nut_warm + NU, eps_safe)),
    "y_sq":         y ** 2,
    "x_mod9":       np.sin(2 * np.pi * x / 9.0),  # periodic hill x-phase
}

feature_names = list(features_dict.keys())
X = np.column_stack([features_dict[k] for k in feature_names])
y_target = A_field.copy()   # continuous target: 0 or best_A

print(f"\nFeatures ({len(feature_names)}): {feature_names}")
print(f"Target: 0 outside SL, {best_A:.4f} in SL  (class balance: {shear_mask.mean():.3f})")

#     Train/val split
rng = np.random.default_rng(42)
idx = rng.permutation(N)
n_val = N // 5   # 20% validation
idx_val, idx_tr = idx[:n_val], idx[n_val:]

X_tr, y_tr = X[idx_tr], y_target[idx_tr]
X_val, y_val = X[idx_val], y_target[idx_val]
sl_tr = shear_mask[idx_tr]; sl_val = shear_mask[idx_val]

print(f"\nTrain: {len(X_tr)}  Val: {len(X_val)}")


feat_mean = X_tr.mean(axis=0)
feat_std  = X_tr.std(axis=0) + 1e-12
X_tr_n  = (X_tr  - feat_mean) / feat_std
X_val_n = (X_val - feat_mean) / feat_std


try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import label_binarize

    y_tr_bin  = (y_tr  > 0).astype(int)
    y_val_bin = (y_val > 0).astype(int)

    print("\n─── Random Forest classifier ───")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_tr_n, y_tr_bin)
    rf_pred_val = rf.predict(X_val_n)
    rf_prob_val = rf.predict_proba(X_val_n)[:, 1]

    print(classification_report(y_val_bin, rf_pred_val,
                                 target_names=["outside_SL", "in_SL"]))
    auc = roc_auc_score(y_val_bin, rf_prob_val)
    print(f"AUC-ROC: {auc:.4f}")

    # Feature importances
    print("\nTop feature importances:")
    imp = list(zip(feature_names, rf.feature_importances_))
    for name, score in sorted(imp, key=lambda t: -t[1])[:8]:
        print(f"  {name:<20} {score:.4f}")

    # Save model
    model_path = OUT_DIR / "kSourceML_rf_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": rf, "feat_mean": feat_mean, "feat_std": feat_std,
                     "feature_names": feature_names, "best_A": best_A,
                     "nu": NU}, f)
    print(f"\nModel saved → {model_path}")


    # Full-domain prediction
    X_all_n = (X - feat_mean) / feat_std
    sl_pred = rf.predict(X_all_n).astype(bool)
    A_pred  = np.where(sl_pred, best_A, 0.0)

    print(f"\nPredicted shear layer: {sl_pred.sum()} cells (true: {shear_mask.sum()})")
    precision = (sl_pred & shear_mask).sum() / max(sl_pred.sum(), 1)
    recall    = (sl_pred & shear_mask).sum() / max(shear_mask.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

    kSourceML_pred   = A_pred  * k_warm
    kSourceML_target = A_field * k_warm
    rmse_src = np.sqrt(np.mean((kSourceML_pred - kSourceML_target)**2))
    print(f"kSourceML RMSE: {rmse_src:.3e} m²/s³")

    # Save predicted field for inspection
    np.savez(OUT_DIR / "kSourceML_predictor_eval.npz",
             cell_xy=cell_xy, A_pred=A_pred, A_target=A_field,
             kSourceML_pred=kSourceML_pred, kSourceML_target=kSourceML_target,
             sl_pred=sl_pred, sl_true=shear_mask)


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

    lines = [of_header("volScalarField", "best", "kSourceML_predicted"),
             "\ndimensions      [0 2 -3 0 0 0 0];\n",
             f"\ninternalField   nonuniform List<scalar>\n{len(kSourceML_pred)}\n(\n"]
    lines.extend(f"{v:.8e}\n" for v in kSourceML_pred)
    lines.append(")\n;\n\nboundaryField\n{\n"
                 "    bottomWall { type zeroGradient; }\n"
                 "    topWall    { type zeroGradient; }\n"
                 "    inlet      { type cyclic; }\n"
                 "    outlet     { type cyclic; }\n"
                 "    defaultFaces { type empty; }\n"
                 "}\n\n// *** //\n")
    pred_field = OUT_DIR / "kSourceML_predicted_rf"
    pred_field.write_text("".join(lines))
    print(f"Predicted field → {pred_field}")

    # Summary stats
    summary = {
        "n_cells": int(N),
        "n_shear_layer_true": int(shear_mask.sum()),
        "n_shear_layer_pred": int(sl_pred.sum()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "auc_roc": float(auc),
        "kSourceML_rmse": float(rmse_src),
        "best_A": float(best_A),
        "feature_names": feature_names,
    }
    with open(OUT_DIR / "predictor_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {OUT_DIR}/predictor_summary.json")

except ImportError as e:
    print(f"sklearn not available: {e}")
    print("Install with: pip install scikit-learn")
