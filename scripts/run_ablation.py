#!/usr/bin/env python3
"""Run ablation experiments and generate figures for Member 3 deliverables.

This script:
1. Runs all 5 ablation configurations.
2. Evaluates building accuracy, floor accuracy, and positioning error.
3. Saves results to results/metrics/ablation_results.csv.
4. Generates comparison bar charts to results/figures/.
5. Generates feature visualisation plots.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_train, load_validation, WAP_COLS, TARGET_COLS, META_COLS
from src.preprocessing import (
    WifiPreprocessor, prepare_features, ABLATION_CONFIGS, get_feature_dictionary
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

FIG = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
METRICS = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")
os.makedirs(FIG, exist_ok=True)
os.makedirs(METRICS, exist_ok=True)

plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading data...")
train_df = load_train()
val_df   = load_validation()
# Note: duplicate removal and empty-scan filtering are applied inside
# prepare_features() as part of the preprocessing pipeline.
print(f"  Train: {train_df.shape}, Validation: {val_df.shape}")

# ── Feature dictionary ───────────────────────────────────────────────────────
fd = get_feature_dictionary()
fd.to_csv(os.path.join(METRICS, "feature_dictionary.csv"), index=False)
print("\nFeature Dictionary:")
print(fd.to_string(index=False))

# ── Ablation experiments ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("ABLATION EXPERIMENTS")
print("="*70)

results = []
for name, config in ABLATION_CONFIGS.items():
    print(f"\n--- Config: {name} ---")
    X_tr, X_va, y_tr, y_va, _, _, pp = prepare_features(
        train_df, val_df, WAP_COLS, TARGET_COLS, META_COLS, config=config
    )
    n_feat = X_tr.shape[1]
    print(f"  Features: {n_feat}")
    
    # Building classification
    rf_bld = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_bld.fit(X_tr, y_tr["BUILDINGID"])
    bld_acc = accuracy_score(y_va["BUILDINGID"], rf_bld.predict(X_va))
    print(f"  Building accuracy: {bld_acc:.4f}")
    
    # Floor classification
    rf_flr = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_flr.fit(X_tr, y_tr["FLOOR"])
    flr_acc = accuracy_score(y_va["FLOOR"], rf_flr.predict(X_va))
    print(f"  Floor accuracy:    {flr_acc:.4f}")
    
    # Position regression (mean Euclidean error)
    rf_lon = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_lat = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_lon.fit(X_tr, y_tr["LONGITUDE"])
    rf_lat.fit(X_tr, y_tr["LATITUDE"])
    pred_lon = rf_lon.predict(X_va)
    pred_lat = rf_lat.predict(X_va)
    pos_err = np.sqrt((pred_lon - y_va["LONGITUDE"].values)**2 + 
                       (pred_lat - y_va["LATITUDE"].values)**2).mean()
    print(f"  Mean pos error:    {pos_err:.2f} m")
    
    results.append({
        "Config": name,
        "Building_Acc": round(bld_acc, 4),
        "Floor_Acc": round(flr_acc, 4),
        "Mean_Pos_Error_m": round(pos_err, 2),
        "n_features": n_feat,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(METRICS, "ablation_results.csv"), index=False)
print("\n" + "="*70)
print("Ablation Results:")
print(results_df.to_string(index=False))
print("="*70)

# ── Ablation bar charts ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(results_df["Config"], results_df["Building_Acc"], color="#2a78d6")
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Building Classification Accuracy")
axes[0].set_ylim(0.9, 1.01)
axes[0].tick_params(axis="x", rotation=30)

axes[1].bar(results_df["Config"], results_df["Floor_Acc"], color="#eb6834")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Floor Classification Accuracy")
axes[1].set_ylim(0.5, 1.0)
axes[1].tick_params(axis="x", rotation=30)

axes[2].bar(results_df["Config"], results_df["Mean_Pos_Error_m"], color="#1baf7a")
axes[2].set_ylabel("Mean Error (m)")
axes[2].set_title("Mean Positioning Error")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "preproc_ablation_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: preproc_ablation_comparison.png")

print("\n✅ All ablation experiments and figures complete!")
