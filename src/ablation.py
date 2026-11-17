"""Ablation study runner for loss functions, regularization and augmentation (Member 3).

Runs the 7 ablation experiments specified in the project guide (Section 4, Step 5):

    Run   Loss                   Dropout   Augmentation
    ───   ────                   ───────   ────────────
    A     Basic combined loss    0.2       None
    B     Uncertainty weighting  0.2       None
    C     Basic combined loss    0.0       None
    D     Basic combined loss    0.3       None
    E     Basic combined loss    0.2       RSSI noise
    F     Basic combined loss    0.2       AP dropout
    G     Uncertainty weighting  0.2       RSSI + AP dropout

Experiments change one factor at a time where possible. Run A is the baseline
(identical to Member 1's multitask model). Each experiment uses the same data
split, preprocessing, and evaluation protocol.

Run from repository root:
    python -m src.ablation
"""

import json
import os
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn

from src.augmentations import APDropout, ComposeAugmentation, RSSINoise
from src.data_preprocessing import prepare_data
from src.dataset import WifiDataset, make_loader
from src.models.loss import BasicCombinedLoss, UncertaintyLoss
from src.models.multitask_net import ALL_TASKS, MultitaskNet
from src.train_multitask import compute_metrics, predict, selection_score, set_seed

# Set writable matplotlib config
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Experiment Configuration
# ──────────────────────────────────────────────────────────────────────────────

ABLATION_EXPERIMENTS = {
    "A": {
        "description": "Baseline: basic combined loss, dropout=0.2, no augmentation",
        "loss_type": "basic",
        "dropout": 0.2,
        "weight_decay": 0.0,
        "augmentation": None,
    },
    "B": {
        "description": "Uncertainty weighting, dropout=0.2, no augmentation",
        "loss_type": "uncertainty",
        "dropout": 0.2,
        "weight_decay": 0.0,
        "augmentation": None,
    },
    "C": {
        "description": "No dropout (dropout=0.0), basic combined loss",
        "loss_type": "basic",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "augmentation": None,
    },
    "D": {
        "description": "Higher dropout (dropout=0.3), basic combined loss",
        "loss_type": "basic",
        "dropout": 0.3,
        "weight_decay": 0.0,
        "augmentation": None,
    },
    "E": {
        "description": "RSSI noise augmentation, basic combined loss, dropout=0.2",
        "loss_type": "basic",
        "dropout": 0.2,
        "weight_decay": 0.0,
        "augmentation": "rssi_noise",
    },
    "F": {
        "description": "AP dropout augmentation, basic combined loss, dropout=0.2",
        "loss_type": "basic",
        "dropout": 0.2,
        "weight_decay": 0.0,
        "augmentation": "ap_dropout",
    },
    "G": {
        "description": "Full: uncertainty weighting + RSSI noise + AP dropout",
        "loss_type": "uncertainty",
        "dropout": 0.2,
        "weight_decay": 0.0,
        "augmentation": "rssi_noise+ap_dropout",
    },
}

# Additional weight decay experiments
WEIGHT_DECAY_EXPERIMENTS = {
    "WD_1e5": {
        "description": "Weight decay 1e-5, basic loss, dropout=0.2",
        "loss_type": "basic",
        "dropout": 0.2,
        "weight_decay": 1e-5,
        "augmentation": None,
    },
    "WD_1e4": {
        "description": "Weight decay 1e-4, basic loss, dropout=0.2",
        "loss_type": "basic",
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "augmentation": None,
    },
}

TRAINING_CONFIG = {
    "hidden": (512, 256, 128),
    "learning_rate": 1e-3,
    "batch_size": 128,
    "max_epochs": 150,
    "patience": 20,
    "coord_loss": "mse",
    "seed": 42,
}


# ──────────────────────────────────────────────────────────────────────────────
# Augmented DataLoader
# ──────────────────────────────────────────────────────────────────────────────

class AugmentedWifiDataset(WifiDataset):
    """WifiDataset that applies augmentations to X during training."""

    def __init__(self, d, augmentation=None):
        super().__init__(d)
        self.augmentation = augmentation

    def __getitem__(self, i):
        x, building, floor, coords = super().__getitem__(i)
        if self.augmentation is not None:
            x = self.augmentation(x)
        return x, building, floor, coords


def make_augmented_loader(d, batch_size, augmentation=None, shuffle=False):
    from torch.utils.data import DataLoader
    return DataLoader(
        AugmentedWifiDataset(d, augmentation=augmentation),
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Training with configurable loss and augmentation
# ──────────────────────────────────────────────────────────────────────────────

def build_augmentation(aug_name):
    """Build augmentation from string specification."""
    if aug_name is None:
        return None
    elif aug_name == "rssi_noise":
        return RSSINoise(std=0.02)
    elif aug_name == "ap_dropout":
        return APDropout(drop_rate=0.10)
    elif aug_name == "rssi_noise+ap_dropout":
        return ComposeAugmentation([RSSINoise(std=0.02), APDropout(drop_rate=0.10)])
    else:
        raise ValueError(f"Unknown augmentation: {aug_name}")


def run_ablation_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over the data with a configurable loss function."""
    training = optimizer is not None
    model.train(training)
    sums, n = {}, 0

    for X, building, floor, coords in loader:
        X = X.to(DEVICE)
        building = building.to(DEVICE)
        floor = floor.to(DEVICE)
        coords = coords.to(DEVICE)

        with torch.set_grad_enabled(training):
            outputs = model(X)
            losses = loss_fn(outputs, building, floor, coords)
            if training:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

        for name, value in losses.items():
            if name.startswith("_"):
                continue  # skip metadata like _log_vars
            sums[name] = sums.get(name, 0.0) + value.item() * len(X)
        n += len(X)

    return {name: total / n for name, total in sums.items()}


def train_ablation_model(data, prep, experiment_name, experiment_config, verbose=True):
    """Train one ablation experiment and return results."""
    config = dict(TRAINING_CONFIG)
    config["dropout"] = experiment_config["dropout"]
    config["weight_decay"] = experiment_config["weight_decay"]

    set_seed(config["seed"])

    # Build model
    model = MultitaskNet(
        n_inputs=data["train"]["X"].shape[1],
        n_buildings=len(prep["building_encoder"].classes_),
        n_floors=len(prep["floor_encoder"].classes_),
        hidden=config["hidden"],
        dropout=config["dropout"],
        tasks=ALL_TASKS,
    ).to(DEVICE)

    # Build loss function
    if experiment_config["loss_type"] == "uncertainty":
        loss_fn = UncertaintyLoss(coord_loss=config["coord_loss"]).to(DEVICE)
        # Include loss parameters in optimizer
        all_params = list(model.parameters()) + list(loss_fn.parameters())
    else:
        loss_fn = BasicCombinedLoss(coord_loss=config["coord_loss"])
        all_params = model.parameters()

    optimizer = torch.optim.Adam(
        all_params, lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    # Build augmentation
    augmentation = build_augmentation(experiment_config["augmentation"])

    # Build data loaders
    train_loader = make_augmented_loader(
        data["train"], config["batch_size"], augmentation=augmentation, shuffle=True
    )
    val_loader = make_loader(data["val"], config["batch_size"])

    # Training loop
    checkpoint_path = MODEL_DIR / f"ablation_{experiment_name}_best.pth"
    best_score, best_epoch = float("inf"), 0
    history = []
    start = time.time()

    for epoch in range(1, config["max_epochs"] + 1):
        train_losses = run_ablation_epoch(model, train_loader, loss_fn, optimizer)
        val_losses = run_ablation_epoch(model, val_loader, loss_fn)
        val_metrics = compute_metrics(predict(model, data["val"], prep), data["val"], prep)
        score = selection_score(val_metrics)

        row = {"epoch": epoch}
        row.update({f"train_{k}_loss": v for k, v in train_losses.items()})
        row.update({f"val_{k}_loss": v for k, v in val_losses.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["val_score"] = score

        # Log uncertainty weights if using uncertainty loss
        if hasattr(loss_fn, "get_effective_weights"):
            weights = loss_fn.get_effective_weights()
            row.update({f"uw_{k}": v for k, v in weights.items()})

        history.append(row)

        if score < best_score:
            best_score, best_epoch = score, epoch
            save_dict = {
                "model_state": model.state_dict(),
                "tasks": ALL_TASKS,
                "config": config,
                "experiment": experiment_config,
                "n_inputs": data["train"]["X"].shape[1],
                "epoch": epoch,
            }
            if hasattr(loss_fn, "state_dict"):
                save_dict["loss_state"] = loss_fn.state_dict()
            torch.save(save_dict, checkpoint_path)

        if verbose and (epoch == 1 or epoch % 20 == 0):
            metric_str = "  ".join(f"{k}={v:.3f}" for k, v in val_metrics.items())
            print(f"  [{experiment_name}] epoch {epoch:3d}  "
                  f"train_loss={train_losses['total']:.4f}  "
                  f"val_score={score:.3f}  {metric_str}")

        if epoch - best_epoch >= config["patience"]:
            break

    # Reload best weights
    best_ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])

    # Final evaluation on val and test
    val_metrics = compute_metrics(predict(model, data["val"], prep), data["val"], prep)
    test_metrics = compute_metrics(predict(model, data["test"], prep), data["test"], prep)

    elapsed = time.time() - start
    if verbose:
        print(f"  [{experiment_name}] DONE: best_epoch={best_epoch}, "
              f"val_score={best_score:.3f}, "
              f"test_bld={test_metrics.get('building_acc', 0)*100:.2f}%, "
              f"test_flr={test_metrics.get('floor_acc', 0)*100:.2f}%, "
              f"test_err={test_metrics.get('mean_error_m', 0):.2f}m, "
              f"time={elapsed:.0f}s\n")

    history_df = pd.DataFrame(history)

    return {
        "experiment": experiment_name,
        "description": experiment_config["description"],
        "loss_type": experiment_config["loss_type"],
        "dropout": experiment_config["dropout"],
        "weight_decay": experiment_config["weight_decay"],
        "augmentation": experiment_config["augmentation"] or "none",
        "best_epoch": best_epoch,
        "total_epochs": len(history),
        "training_time_s": round(elapsed, 1),
        "val_building_acc": val_metrics.get("building_acc"),
        "val_floor_acc": val_metrics.get("floor_acc"),
        "val_joint_acc": val_metrics.get("joint_acc"),
        "val_mean_error_m": val_metrics.get("mean_error_m"),
        "val_median_error_m": val_metrics.get("median_error_m"),
        "val_p90_error_m": val_metrics.get("p90_error_m"),
        "val_score": best_score,
        "test_building_acc": test_metrics.get("building_acc"),
        "test_floor_acc": test_metrics.get("floor_acc"),
        "test_joint_acc": test_metrics.get("joint_acc"),
        "test_mean_error_m": test_metrics.get("mean_error_m"),
        "test_median_error_m": test_metrics.get("median_error_m"),
        "test_p90_error_m": test_metrics.get("p90_error_m"),
        "test_score": selection_score(test_metrics),
    }, history_df, model


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_ablation_results(results_df, save_dir):
    """Generate comparison plots for the ablation study."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    plt.style.use(style)

    # ── 1. Grouped bar chart: Building Acc, Floor Acc, Mean Error ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)

    experiments = results_df["experiment"].values
    x = np.arange(len(experiments))
    width = 0.6

    # Building accuracy
    ax = axes[0]
    vals = results_df["test_building_acc"].values * 100
    bars = ax.bar(x, vals, width, color="#2196F3", alpha=0.85, edgecolor="white")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Building Classification Accuracy", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=9)
    ax.set_ylim(min(vals) - 1, 100.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5)

    # Floor accuracy
    ax = axes[1]
    vals = results_df["test_floor_acc"].values * 100
    bars = ax.bar(x, vals, width, color="#4CAF50", alpha=0.85, edgecolor="white")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Floor Classification Accuracy", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=9)
    ax.set_ylim(min(vals) - 3, max(vals) + 3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5)

    # Mean coordinate error
    ax = axes[2]
    vals = results_df["test_mean_error_m"].values
    bars = ax.bar(x, vals, width, color="#FF9800", alpha=0.85, edgecolor="white")
    ax.set_ylabel("Mean Error (m)", fontweight="bold")
    ax.set_title("Mean Coordinate Error", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}m", ha="center", va="bottom", fontsize=7.5)

    plt.suptitle("Ablation Study: Loss Functions, Regularization & Augmentation",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "ablation_bar_comparison.png", bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved ablation bar comparison to {save_dir / 'ablation_bar_comparison.png'}")

    # ── 2. EvAAL Score comparison (lower is better) ──
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
              "#7f7f7f", "#bcbd22"]
    bars = ax.barh(experiments[::-1], results_df["test_score"].values[::-1],
                   color=colors[:len(experiments)][::-1], alpha=0.85, edgecolor="white", height=0.6)
    for bar, v in zip(bars, results_df["test_score"].values[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", ha="left", va="center", fontsize=9)
    ax.set_xlabel("EvAAL Score (lower is better)", fontweight="bold")
    ax.set_title("Ablation Study: EvAAL Competition Score Comparison", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "ablation_evaal_scores.png", bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved EvAAL score comparison to {save_dir / 'ablation_evaal_scores.png'}")

    # ── 3. Training curves for selected experiments ──
    return


def plot_training_curves(histories, save_dir):
    """Plot training/validation loss curves for each experiment."""
    save_dir = Path(save_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

    for name, hist_df in histories.items():
        epochs = hist_df["epoch"].values

        # Total loss
        axes[0, 0].plot(epochs, hist_df["train_total_loss"], label=f"{name} train", alpha=0.7)
        axes[0, 0].plot(epochs, hist_df["val_total_loss"], label=f"{name} val",
                        linestyle="--", alpha=0.7)

        # Building loss
        if "train_building_loss" in hist_df.columns:
            axes[0, 1].plot(epochs, hist_df["val_building_loss"], label=name, alpha=0.8)

        # Floor accuracy
        if "val_floor_acc" in hist_df.columns:
            axes[1, 0].plot(epochs, hist_df["val_floor_acc"] * 100, label=name, alpha=0.8)

        # Mean coordinate error
        if "val_mean_error_m" in hist_df.columns:
            axes[1, 1].plot(epochs, hist_df["val_mean_error_m"], label=name, alpha=0.8)

    axes[0, 0].set_title("Total Loss (Train vs Val)", fontweight="bold")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(fontsize=6, ncol=2)

    axes[0, 1].set_title("Validation Building Loss", fontweight="bold")
    axes[0, 1].set_ylabel("Cross-Entropy Loss")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].set_title("Validation Floor Accuracy", fontweight="bold")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].set_title("Validation Mean Coordinate Error", fontweight="bold")
    axes[1, 1].set_ylabel("Error (m)")
    axes[1, 1].legend(fontsize=7)

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves Across Ablation Experiments", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "ablation_training_curves.png", bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved training curves to {save_dir / 'ablation_training_curves.png'}")


def plot_uncertainty_weights(histories, save_dir):
    """Plot how the learned uncertainty weights evolve during training for experiment B and G."""
    save_dir = Path(save_dir)
    uncertainty_exps = {k: v for k, v in histories.items() if "uw_building_weight" in v.columns}

    if not uncertainty_exps:
        print("[Plot] No uncertainty weight data found, skipping.")
        return

    fig, axes = plt.subplots(1, len(uncertainty_exps), figsize=(7 * len(uncertainty_exps), 5), dpi=200)
    if len(uncertainty_exps) == 1:
        axes = [axes]

    for ax, (name, hist_df) in zip(axes, uncertainty_exps.items()):
        epochs = hist_df["epoch"].values
        ax.plot(epochs, hist_df["uw_building_weight"], label="Building", linewidth=2)
        ax.plot(epochs, hist_df["uw_floor_weight"], label="Floor", linewidth=2)
        ax.plot(epochs, hist_df["uw_coords_weight"], label="Coordinates", linewidth=2)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Uniform weight")
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Effective Weight exp(-s_i)", fontweight="bold")
        ax.set_title(f"Experiment {name}: Learned Task Weights", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "ablation_uncertainty_weights.png", bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved uncertainty weight evolution to {save_dir / 'ablation_uncertainty_weights.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all_ablations(include_weight_decay=True):
    """Run all ablation experiments and save results."""
    start_all = time.time()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("MEMBER 3: ABLATION STUDY — LOSS FUNCTIONS, REGULARIZATION & AUGMENTATION")
    print("=" * 78)

    # Load data (same split as Member 1 and 2)
    print("\n[Data] Loading and preprocessing data splits...")
    data, prep = prepare_data(save=False)
    print(f"  Train: {len(data['train']['X'])}, Val: {len(data['val']['X'])}, "
          f"Test: {len(data['test']['X'])}, Features: {data['train']['X'].shape[1]}")

    # Combine all experiments
    all_experiments = dict(ABLATION_EXPERIMENTS)
    if include_weight_decay:
        all_experiments.update(WEIGHT_DECAY_EXPERIMENTS)

    results = []
    histories = {}

    for exp_name, exp_config in all_experiments.items():
        print(f"\n{'─' * 78}")
        print(f"EXPERIMENT {exp_name}: {exp_config['description']}")
        print(f"{'─' * 78}")

        result, history_df, model = train_ablation_model(
            data, prep, exp_name, exp_config, verbose=True
        )
        results.append(result)
        histories[exp_name] = history_df

        # Save individual training history
        history_df.to_csv(METRICS_DIR / f"ablation_{exp_name}_history.csv", index=False)

    # ── Save master results ──
    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_DIR / "ablation_results_member3.csv", index=False)
    print(f"\n[Save] Ablation results saved to {METRICS_DIR / 'ablation_results_member3.csv'}")

    # ── Save regularization comparison (subset) ──
    reg_rows = results_df[results_df["experiment"].isin(["A", "C", "D", "WD_1e5", "WD_1e4"])]
    if len(reg_rows) > 0:
        reg_rows.to_csv(METRICS_DIR / "regularization_comparison.csv", index=False)
        print(f"[Save] Regularization comparison saved to {METRICS_DIR / 'regularization_comparison.csv'}")

    # ── Print summary table ──
    print("\n" + "=" * 78)
    print("ABLATION STUDY RESULTS (TEST SET)")
    print("=" * 78)
    summary_cols = ["experiment", "loss_type", "dropout", "augmentation",
                    "test_building_acc", "test_floor_acc", "test_mean_error_m",
                    "test_median_error_m", "test_score"]
    summary = results_df[summary_cols].copy()
    summary["test_building_acc"] = (summary["test_building_acc"] * 100).round(2).astype(str) + "%"
    summary["test_floor_acc"] = (summary["test_floor_acc"] * 100).round(2).astype(str) + "%"
    for col in ["test_mean_error_m", "test_median_error_m", "test_score"]:
        summary[col] = summary[col].round(3)
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 15)
    print(summary.to_string(index=False))

    # ── Identify best experiment ──
    best_idx = results_df["test_score"].idxmin()
    best = results_df.iloc[best_idx]
    print(f"\n★ Best experiment: {best['experiment']} ({best['description']})")
    print(f"  EvAAL Score: {best['test_score']:.3f}")

    # ── Generate plots ──
    print("\n[Plotting] Generating comparison figures...")
    plot_ablation_results(results_df, FIGURES_DIR)
    plot_training_curves(histories, FIGURES_DIR)
    plot_uncertainty_weights(histories, FIGURES_DIR)

    # ── Save run metadata ──
    run_info = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "device": str(DEVICE),
        "total_runtime_s": round(time.time() - start_all, 2),
        "num_experiments": len(results),
        "best_experiment": best["experiment"],
        "best_test_score": round(best["test_score"], 4),
        "data_sizes": {k: len(v["X"]) for k, v in data.items()},
    }
    with open(METRICS_DIR / "ablation_run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    total_time = time.time() - start_all
    print(f"\n{'=' * 78}")
    print(f"All {len(results)} ablation experiments completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 78}")

    return results_df


if __name__ == "__main__":
    run_all_ablations()
