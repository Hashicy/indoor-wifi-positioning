"""Master training, benchmarking, and evaluation script for Hybrid Deep Learning (Member 2).

Executes the four core experiments defined in the Phase 2 specification:
1. Audit & Verification of Classical Baselines (Random Forest, Extra Trees, XGBoost)
2. Experiment A : Deep Learning Only (Member 1 Multitask Net baseline)
3. Experiment B : DL + Tree Probability Stacking (Hierarchical Structural Priors)
4. Experiment C : DL + Residual Coordinate Correction (Gradient Boosted Spatial Residuals)
5. Full Hybrid  : Stacked Multitask DL + Residual Coordinate Correction

Run with:
    .conda/bin/python -m src.train_hybrid
"""

import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_preprocessing import prepare_data
from src.dataset import make_loader
from src.models.hybrid_model import HybridPositioningSystem
from src.models.multitask_net import ALL_TASKS, MultitaskNet
from src.residual_correction import ResidualCoordinateCorrector
from src.stacking_features import TreeProbabilityExtractor, concatenate_hybrid_features
from src.train_multitask import (
    CONFIG,
    DEVICE,
    compute_losses,
    compute_metrics,
    predict,
    run_epoch,
    selection_score,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

# Set writable directory for matplotlib cache
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"


# ------------------------------------------------------------------------------
# Classical Baseline Audit
# ------------------------------------------------------------------------------
def audit_classical_baseline() -> pd.DataFrame:
    """Load and verify the Phase 1 Classical ML baseline summary."""
    baseline_csv = METRICS_DIR / "classical_ml_summary.csv"
    if not baseline_csv.exists():
        print("[Audit] classical_ml_summary.csv not found; skipping baseline audit.")
        return pd.DataFrame()

    df = pd.read_csv(baseline_csv)
    print("\n" + "=" * 78)
    print("STEP 1: AUDIT OF EXISTING CLASSICAL ML BASELINES (PHASE 1)")
    print("=" * 78)
    for _, row in df.iterrows():
        print(f"  {row['Model']:<28} | Bld Acc: {row['Building Acc']*100:.2f}% | "
              f"Flr Acc: {row['Floor Acc']*100:.2f}% | Mean Err: {row['Mean Error (m)']:.2f}m | "
              f"Median Err: {row['Median Error (m)']:.2f}m")
    print("=" * 78 + "\n")
    return df


# ------------------------------------------------------------------------------
# Helper Evaluation
# ------------------------------------------------------------------------------
def evaluate_hybrid_predictions(
    pred_coords_m: np.ndarray,
    pred_bld: np.ndarray,
    pred_flr: np.ndarray,
    data_split: Dict[str, Any],
    prep: Dict[str, Any],
) -> Dict[str, float]:
    """Calculate all standard benchmark metrics for a set of predictions."""
    true_bld = prep["building_encoder"].inverse_transform(data_split["building"])
    true_flr = prep["floor_encoder"].inverse_transform(data_split["floor"])
    true_coords_m = data_split["coords_m"]

    bld_acc = float(np.mean(pred_bld == true_bld))
    flr_acc = float(np.mean(pred_flr == true_flr))
    flr_abs_diff = float(np.mean(np.abs(pred_flr - true_flr)))
    joint_acc = float(np.mean((pred_bld == true_bld) & (pred_flr == true_flr)))

    errors = np.linalg.norm(pred_coords_m - true_coords_m, axis=1)
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    p75_err = float(np.percentile(errors, 75))
    p90_err = float(np.percentile(errors, 90))
    mae = float(np.mean(np.abs(pred_coords_m - true_coords_m)))
    rmse = float(np.sqrt(np.mean((pred_coords_m - true_coords_m) ** 2)))

    evaal_score = mean_err + 50.0 * (1.0 - bld_acc) + 4.0 * flr_abs_diff

    return {
        "building_acc": bld_acc,
        "floor_acc": flr_acc,
        "floor_abs_diff": flr_abs_diff,
        "joint_acc": joint_acc,
        "mean_error_m": mean_err,
        "median_error_m": median_err,
        "p75_error_m": p75_err,
        "p90_error_m": p90_err,
        "coord_mae_m": mae,
        "coord_rmse_m": rmse,
        "evaal_score": evaal_score,
    }


# ------------------------------------------------------------------------------
# Train Stacked Multitask Model
# ------------------------------------------------------------------------------
def train_stacked_multitask_network(
    data_hybrid: Dict[str, Dict[str, Any]],
    prep: Dict[str, Any],
    config: Dict[str, Any] = CONFIG,
    name: str = "hybrid_multitask_stacked",
) -> Tuple[MultitaskNet, pd.DataFrame]:
    """Train MultitaskNet on the stacked hybrid feature space [RSSI, P(Bld), P(Flr)]."""
    set_seed(config["seed"])
    n_inputs = data_hybrid["train"]["X"].shape[1]

    model = MultitaskNet(
        n_inputs=n_inputs,
        n_buildings=len(prep["building_encoder"].classes_),
        n_floors=len(prep["floor_encoder"].classes_),
        hidden=config["hidden"],
        dropout=config["dropout"],
        tasks=ALL_TASKS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    train_loader = make_loader(data_hybrid["train"], config["batch_size"], shuffle=True)
    val_loader = make_loader(data_hybrid["val"], config["batch_size"])

    checkpoint_path = MODEL_DIR / f"{name}_best.pth"
    best_score, best_epoch, history = float("inf"), 0, []
    start_time = time.time()

    print(f"\n[Training] Starting Stacked MultitaskNet (inputs={n_inputs}, device={DEVICE})...")
    for epoch in range(1, config["max_epochs"] + 1):
        train_losses = run_epoch(model, train_loader, config, optimizer)
        val_losses = run_epoch(model, val_loader, config)

        val_pred = predict(model, data_hybrid["val"], prep)
        val_metrics = compute_metrics(val_pred, data_hybrid["val"], prep)
        score = selection_score(val_metrics)

        row = {"epoch": epoch}
        row.update({f"train_{k}_loss": v for k, v in train_losses.items()})
        row.update({f"val_{k}_loss": v for k, v in val_losses.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["val_score"] = score
        history.append(row)

        if score < best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tasks": ALL_TASKS,
                    "config": config,
                    "n_inputs": n_inputs,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Epoch {epoch:3d}/{config['max_epochs']} | Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | Val Score: {score:.3f} | "
                f"Bld: {val_metrics.get('building_acc', 0)*100:.1f}% | "
                f"Flr: {val_metrics.get('floor_acc', 0)*100:.1f}% | "
                f"Mean Err: {val_metrics.get('mean_error_m', 0):.2f}m"
            )

        if epoch - best_epoch >= config["patience"]:
            print(f"  Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}).")
            break

    # Reload best weights
    best_ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])
    history_df = pd.DataFrame(history)
    history_df.to_csv(METRICS_DIR / f"{name}_training_history.csv", index=False)
    print(f"[Training] Completed in {time.time()-start_time:.1f}s | Best Val Score: {best_score:.3f}")
    return model, history_df


# ------------------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------------------
def generate_hybrid_plots(
    error_dict: Dict[str, np.ndarray],
    save_dir: Path = FIGURES_DIR,
):
    """Generate publication-ready CDF and Boxplot comparison figures."""
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Error CDF Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    palette = {
        "Classical RF (Phase 1)": "#7f7f7f",
        "Experiment A: DL Baseline": "#1f77b4",
        "Experiment B: DL + Stacking": "#ff7f0e",
        "Experiment C: DL + Residual GBDT": "#2ca02c",
        "Full Hybrid (Stacked DL + GBDT)": "#d62728",
    }
    linestyles = {
        "Classical RF (Phase 1)": ":",
        "Experiment A: DL Baseline": "--",
        "Experiment B: DL + Stacking": "-.",
        "Experiment C: DL + Residual GBDT": "--",
        "Full Hybrid (Stacked DL + GBDT)": "-",
    }

    max_x = 35.0
    for name, errors in error_dict.items():
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        color = palette.get(name, "#333333")
        ls = linestyles.get(name, "-")
        lw = 2.4 if "Full Hybrid" in name else 1.8
        ax.plot(sorted_errors, cdf * 100.0, label=name, color=color, linestyle=ls, linewidth=lw)

    ax.axvline(5.0, color="gray", linestyle=":", alpha=0.6, label="5 m Threshold")
    ax.axvline(10.0, color="gray", linestyle="--", alpha=0.6, label="10 m Threshold")
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, 100)
    ax.set_xlabel("2D Euclidean Localization Error (metres)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Percentage of Scans (%)", fontsize=11, fontweight="bold")
    ax.set_title("Empirical Error CDF: Classical vs. Deep Learning vs. Hybrid Models", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    cdf_path = save_dir / "hybrid_comparison_cdf.png"
    plt.savefig(cdf_path)
    plt.close()
    print(f"[Plot] Saved CDF comparison to {cdf_path}")

    # 2. Error Boxplot
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    data_to_plot = list(error_dict.values())
    labels = list(error_dict.keys())
    box = ax.boxplot(
        data_to_plot,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
    )
    colors = ["#d9d9d9", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_ylabel("2D Error (metres)", fontsize=11, fontweight="bold")
    ax.set_title("Test Coordinate Error Distribution Across Architectures (Outliers Hidden)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    box_path = save_dir / "hybrid_error_boxplot.png"
    plt.savefig(box_path)
    plt.close()
    print(f"[Plot] Saved Error Boxplot to {box_path}")


# ------------------------------------------------------------------------------
# Main Experiment Runner
# ------------------------------------------------------------------------------
def run_hybrid_experiments():
    start_all = time.time()
    set_seed(42)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Audit Classical Baseline
    audit_df = audit_classical_baseline()

    # 2. Load and Prepare Leakage-Free Dataset Split
    print("[Data] Loading and preprocessing data splits...")
    data, prep = prepare_data(save=False)
    X_train, y_bld_train, y_flr_train = data["train"]["X"], data["train"]["building"], data["train"]["floor"]
    X_val, y_bld_val, y_flr_val = data["val"]["X"], data["val"]["building"], data["val"]["floor"]
    X_test, y_bld_test, y_flr_test = data["test"]["X"], data["test"]["building"], data["test"]["floor"]

    true_train_m = data["train"]["coords_m"]
    true_val_m = data["val"]["coords_m"]
    true_test_m = data["test"]["coords_m"]

    # --------------------------------------------------------------------------
    # EXPERIMENT A: Baseline Multitask DL Model (Member 1)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT A: DEEP LEARNING ONLY (MULTITASK NET BASELINE)")
    print("=" * 78)
    dl_baseline = MultitaskNet(n_inputs=X_train.shape[1], tasks=ALL_TASKS).to(DEVICE)
    dl_ckpt_path = MODEL_DIR / "multitask_best.pth"
    if dl_ckpt_path.exists():
        state = torch.load(dl_ckpt_path, map_location=DEVICE)
        dl_baseline.load_state_dict(state["model_state"])
        print(f"Loaded existing MultitaskNet checkpoint from {dl_ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint {dl_ckpt_path} not found. Please train Member 1 model first.")

    pred_train_dl = predict(dl_baseline, data["train"], prep)
    pred_val_dl = predict(dl_baseline, data["val"], prep)
    pred_test_dl = predict(dl_baseline, data["test"], prep)

    m_dl_val = evaluate_hybrid_predictions(pred_val_dl["coords_m"], pred_val_dl["building"], pred_val_dl["floor"], data["val"], prep)
    m_dl_test = evaluate_hybrid_predictions(pred_test_dl["coords_m"], pred_test_dl["building"], pred_test_dl["floor"], data["test"], prep)

    print(f"  DL Test: Bld={m_dl_test['building_acc']*100:.2f}% | Flr={m_dl_test['floor_acc']*100:.2f}% | "
          f"Mean Err={m_dl_test['mean_error_m']:.2f}m | Median Err={m_dl_test['median_error_m']:.2f}m | "
          f"P90={m_dl_test['p90_error_m']:.2f}m")

    # --------------------------------------------------------------------------
    # EXPERIMENT B: DL + Tree Probability Stacking
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT B: HYBRID STACKING (DL + TREE CLASS PROBABILITIES)")
    print("=" * 78)
    print("Fitting TreeProbabilityExtractor with 5-fold Out-Of-Fold generation on Train...")
    tree_extractor = TreeProbabilityExtractor(
        n_estimators=100,
        min_samples_leaf=1,
        n_splits=5,
        random_state=42,
        n_jobs=-1,
    )
    oof_bld_train, oof_flr_train = tree_extractor.fit_transform_train(X_train, y_bld_train, y_flr_train)
    val_bld_p, val_flr_p = tree_extractor.transform(X_val)
    test_bld_p, test_flr_p = tree_extractor.transform(X_test)
    tree_extractor.save()

    print(f"  OOF Train Tree Accuracy : Building={np.mean(oof_bld_train.argmax(1) == y_bld_train)*100:.2f}% | "
          f"Floor={np.mean(oof_flr_train.argmax(1) == y_flr_train)*100:.2f}%")
    print(f"  Test Tree Accuracy      : Building={np.mean(test_bld_p.argmax(1) == y_bld_test)*100:.2f}% | "
          f"Floor={np.mean(test_flr_p.argmax(1) == y_flr_test)*100:.2f}%")

    X_train_stacked = concatenate_hybrid_features(X_train, oof_bld_train, oof_flr_train)
    X_val_stacked = concatenate_hybrid_features(X_val, val_bld_p, val_flr_p)
    X_test_stacked = concatenate_hybrid_features(X_test, test_bld_p, test_flr_p)

    data_stacked = {
        "train": {**data["train"], "X": X_train_stacked},
        "val": {**data["val"], "X": X_val_stacked},
        "test": {**data["test"], "X": X_test_stacked},
    }

    stacked_cfg = {**CONFIG, "max_epochs": 120, "patience": 20}
    stacked_dl_model, _ = train_stacked_multitask_network(
        data_stacked, prep, config=stacked_cfg, name="hybrid_multitask_stacked"
    )

    pred_train_stacked = predict(stacked_dl_model, data_stacked["train"], prep)
    pred_val_stacked = predict(stacked_dl_model, data_stacked["val"], prep)
    pred_test_stacked = predict(stacked_dl_model, data_stacked["test"], prep)

    m_stack_val = evaluate_hybrid_predictions(pred_val_stacked["coords_m"], pred_val_stacked["building"], pred_val_stacked["floor"], data["val"], prep)
    m_stack_test = evaluate_hybrid_predictions(pred_test_stacked["coords_m"], pred_test_stacked["building"], pred_test_stacked["floor"], data["test"], prep)

    print(f"  Stacked Test: Bld={m_stack_test['building_acc']*100:.2f}% | Flr={m_stack_test['floor_acc']*100:.2f}% | "
          f"Mean Err={m_stack_test['mean_error_m']:.2f}m | Median Err={m_stack_test['median_error_m']:.2f}m | "
          f"P90={m_stack_test['p90_error_m']:.2f}m")

    # --------------------------------------------------------------------------
    # EXPERIMENT C: DL + Residual Gradient Boosting
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT C: HYBRID RESIDUAL CORRECTION (DL BASELINE + GBDT RESIDUALS)")
    print("=" * 78)
    residual_corrector_c = ResidualCoordinateCorrector(
        max_iter=100,
        max_depth=6,
        learning_rate=0.04,
        l2_regularization=2.0,
        use_pred_coords_as_features=True,
        random_state=42,
    )
    residual_corrector_c.fit(X_train, true_train_m, pred_train_dl["coords_m"])
    residual_corrector_c.save(MODEL_DIR / "residual_booster_c.joblib")

    coords_val_res_c = residual_corrector_c.correct_coordinates(X_val, pred_val_dl["coords_m"])
    coords_test_res_c = residual_corrector_c.correct_coordinates(X_test, pred_test_dl["coords_m"])

    m_res_val = evaluate_hybrid_predictions(coords_val_res_c, pred_val_dl["building"], pred_val_dl["floor"], data["val"], prep)
    m_res_test = evaluate_hybrid_predictions(coords_test_res_c, pred_test_dl["building"], pred_test_dl["floor"], data["test"], prep)

    print(f"  Residual Test: Bld={m_res_test['building_acc']*100:.2f}% | Flr={m_res_test['floor_acc']*100:.2f}% | "
          f"Mean Err={m_res_test['mean_error_m']:.2f}m | Median Err={m_res_test['median_error_m']:.2f}m | "
          f"P90={m_res_test['p90_error_m']:.2f}m")

    # --------------------------------------------------------------------------
    # EXPERIMENT D: Full Combined Hybrid (Stacked DL + Residual GBDT)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("EXPERIMENT D: FULL COMBINED HYBRID (STACKED DL + RESIDUAL GBDT)")
    print("=" * 78)
    residual_corrector_full = ResidualCoordinateCorrector(
        max_iter=100,
        max_depth=6,
        learning_rate=0.04,
        l2_regularization=2.0,
        use_pred_coords_as_features=True,
        random_state=42,
    )
    residual_corrector_full.fit(X_train_stacked, true_train_m, pred_train_stacked["coords_m"])
    residual_corrector_full.save(MODEL_DIR / "residual_booster_full.joblib")

    coords_val_full = residual_corrector_full.correct_coordinates(X_val_stacked, pred_val_stacked["coords_m"])
    coords_test_full = residual_corrector_full.correct_coordinates(X_test_stacked, pred_test_stacked["coords_m"])

    m_full_val = evaluate_hybrid_predictions(coords_val_full, pred_val_stacked["building"], pred_val_stacked["floor"], data["val"], prep)
    m_full_test = evaluate_hybrid_predictions(coords_test_full, pred_test_stacked["building"], pred_test_stacked["floor"], data["test"], prep)

    print(f"  Full Hybrid Test: Bld={m_full_test['building_acc']*100:.2f}% | Flr={m_full_test['floor_acc']*100:.2f}% | "
          f"Mean Err={m_full_test['mean_error_m']:.2f}m | Median Err={m_full_test['median_error_m']:.2f}m | "
          f"P90={m_full_test['p90_error_m']:.2f}m")

    # Save Unified Hybrid Positioning System
    hybrid_system = HybridPositioningSystem(
        mode="full_hybrid",
        dl_model=stacked_dl_model,
        tree_extractor=tree_extractor,
        residual_corrector=residual_corrector_full,
    )
    hybrid_system.save(MODEL_DIR / "hybrid_system_best.joblib")
    print(f"\n[Artifact] Saved end-to-end HybridPositioningSystem to {MODEL_DIR / 'hybrid_system_best.joblib'}")

    # --------------------------------------------------------------------------
    # Master Comparison Table Construction
    # --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("MASTER HYBRID COMPARISON TABLE")
    print("=" * 78)

    records = [
        {"Model": "Experiment A: DL Baseline", "Split": "val", **m_dl_val},
        {"Model": "Experiment A: DL Baseline", "Split": "test", **m_dl_test},
        {"Model": "Experiment B: DL + Tree Stacking", "Split": "val", **m_stack_val},
        {"Model": "Experiment B: DL + Tree Stacking", "Split": "test", **m_stack_test},
        {"Model": "Experiment C: DL + Residual GBDT", "Split": "val", **m_res_val},
        {"Model": "Experiment C: DL + Residual GBDT", "Split": "test", **m_res_test},
        {"Model": "Full Hybrid: Stacked DL + GBDT", "Split": "val", **m_full_val},
        {"Model": "Full Hybrid: Stacked DL + GBDT", "Split": "test", **m_full_test},
    ]
    df_results = pd.DataFrame(records)
    csv_out = METRICS_DIR / "hybrid_comparison.csv"
    df_results.to_csv(csv_out, index=False)
    print(f"Saved master comparison CSV to {csv_out}")

    # Print clean formatted markdown table for test split
    test_view = df_results[df_results["Split"] == "test"][[
        "Model", "building_acc", "floor_acc", "joint_acc",
        "mean_error_m", "median_error_m", "p90_error_m", "evaal_score"
    ]].copy()
    test_view["building_acc"] = (test_view["building_acc"] * 100).round(2).astype(str) + "%"
    test_view["floor_acc"] = (test_view["floor_acc"] * 100).round(2).astype(str) + "%"
    test_view["joint_acc"] = (test_view["joint_acc"] * 100).round(2).astype(str) + "%"
    for col in ["mean_error_m", "median_error_m", "p90_error_m", "evaal_score"]:
        test_view[col] = test_view[col].round(2)
    print("\nTest Split Benchmark Summary:")
    print(test_view.to_string(index=False))

    # --------------------------------------------------------------------------
    # Generate Plots
    # --------------------------------------------------------------------------
    rf_errors = None
    if (METRICS_DIR / "validation_predictions.csv").exists():
        val_pred_df = pd.read_csv(METRICS_DIR / "validation_predictions.csv")
        if "pred_x_rf" in val_pred_df.columns and "pred_y_rf" in val_pred_df.columns:
            true_xy = val_pred_df[["LONGITUDE", "LATITUDE"]].to_numpy()
            pred_xy = val_pred_df[["pred_x_rf", "pred_y_rf"]].to_numpy()
            rf_errors = np.linalg.norm(true_xy - pred_xy, axis=1)

    error_dict = {
        "Experiment A: DL Baseline": np.linalg.norm(true_test_m - pred_test_dl["coords_m"], axis=1),
        "Experiment B: DL + Stacking": np.linalg.norm(true_test_m - pred_test_stacked["coords_m"], axis=1),
        "Experiment C: DL + Residual GBDT": np.linalg.norm(true_test_m - coords_test_res_c, axis=1),
        "Full Hybrid (Stacked DL + GBDT)": np.linalg.norm(true_test_m - coords_test_full, axis=1),
    }
    if rf_errors is not None:
        error_dict = {"Classical RF (Phase 1)": rf_errors, **error_dict}

    generate_hybrid_plots(error_dict, FIGURES_DIR)

    # Save run metadata
    run_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(DEVICE),
        "total_runtime_s": round(time.time() - start_all, 2),
        "best_hybrid_test_mean_error_m": round(m_full_test["mean_error_m"], 3),
        "best_hybrid_test_median_error_m": round(m_full_test["median_error_m"], 3),
        "best_hybrid_test_floor_acc": round(m_full_test["floor_acc"], 4),
        "best_hybrid_test_building_acc": round(m_full_test["building_acc"], 4),
    }
    with open(METRICS_DIR / "hybrid_run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\nAll experiments successfully completed in {time.time() - start_all:.1f}s.")
    return df_results


if __name__ == "__main__":
    run_hybrid_experiments()
