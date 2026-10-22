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

# ── Feature visualisations (using full pipeline, unscaled for interpretability)
print("\nGenerating feature visualisation plots...")
X_tr_vis, X_va_vis, y_tr, y_va, _, _, _ = prepare_features(
    train_df, val_df, WAP_COLS, TARGET_COLS, META_COLS,
    config={"impute": True, "indicators": True, "aggregates": True, "scale": False}
)

plot_df = X_tr_vis[["num_aps", "max_rssi", "mean_rssi", "std_rssi", "range_rssi"]].copy()
plot_df["BUILDINGID"] = y_tr["BUILDINGID"].values
plot_df["FLOOR"] = y_tr["FLOOR"].values

# 1. num_aps by building
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=plot_df, x="BUILDINGID", y="num_aps", palette="Set2", ax=ax)
ax.set_xlabel("Building ID")
ax.set_ylabel("Number of Detected APs")
ax.set_title("Detected Access Points per Scan by Building")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "preproc_numaps_by_building.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: preproc_numaps_by_building.png")

# 2. max_rssi by floor
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=plot_df, x="FLOOR", y="max_rssi", palette="Set2", ax=ax)
ax.set_xlabel("Floor")
ax.set_ylabel("Strongest RSSI (dBm)")
ax.set_title("Strongest Signal per Scan by Floor")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "preproc_maxrssi_by_floor.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: preproc_maxrssi_by_floor.png")

# 3. mean_rssi vs num_aps colored by building
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
for bld in [0, 1, 2]:
    mask = plot_df["BUILDINGID"] == bld
    ax.scatter(plot_df.loc[mask, "num_aps"], plot_df.loc[mask, "mean_rssi"],
               alpha=0.3, s=10, c=colors[bld], label=f"Building {bld}")
ax.set_xlabel("Number of Detected APs")
ax.set_ylabel("Mean RSSI (dBm)")
ax.set_title("Mean RSSI vs. Detected APs (coloured by Building)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "preproc_meanrssi_vs_numaps.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: preproc_meanrssi_vs_numaps.png")

# 4. RSSI distribution after imputation
fig, ax = plt.subplots(figsize=(8, 5))
active_waps = [c for c in X_tr_vis.columns if c.startswith("WAP") and "_miss" not in c]
all_rssi = X_tr_vis[active_waps].values.flatten()
observed = all_rssi[all_rssi != -110]
ax.hist(observed, bins=80, color="#2a78d6", edgecolor="white", linewidth=0.3)
ax.set_xlabel("RSSI (dBm)")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Observed RSSI Values (after preprocessing)")
ax.axvline(x=-110, color="red", linestyle="--", alpha=0.7, label="Impute value (-110)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "preproc_rssi_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: preproc_rssi_distribution.png")

# 5. Heatmap: AP detection rate by building (top 50 most-detected APs)
print("Generating AP detection heatmap...")
wap_subset = [c for c in WAP_COLS if c in X_tr_vis.columns]
detection = (X_tr_vis[wap_subset] != -110).astype(int)
detection["BUILDINGID"] = y_tr["BUILDINGID"].values
det_by_bld = detection.groupby("BUILDINGID")[wap_subset].mean()

# Pick top 50 APs by overall detection rate
overall = det_by_bld.mean()
top50 = overall.nlargest(50).index.tolist()

fig, ax = plt.subplots(figsize=(16, 4))
sns.heatmap(det_by_bld[top50], ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
            xticklabels=True, yticklabels=["Bld 0", "Bld 1", "Bld 2"])
ax.set_title("AP Detection Rate by Building (Top 50 APs)")
ax.set_xlabel("Access Point")
plt.xticks(fontsize=6, rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "preproc_ap_detection_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: preproc_ap_detection_heatmap.png")

print("\n✅ All ablation experiments and figures complete!")
