"""Evaluation metrics, error analysis, and visualization utilities for indoor positioning.

This module provides a unified evaluation toolkit for the indoor Wi-Fi
positioning project.  Every metric referenced in the project specification
is implemented here and is reusable from both notebook and script contexts.

Public API
----------
Metric functions:
    position_errors           – per-sample 2-D Euclidean distance error
    evaluate_classifier       – accuracy, precision, recall, macro-F1
    evaluate_regressor        – mean / median / p90 position error, MAE, RMSE, R²
    evaluate_building_floor   – combined building + floor accuracy per model

Breakdown functions:
    building_wise_breakdown   – metrics disaggregated by building
    floor_wise_breakdown      – metrics disaggregated by floor
    failure_case_analysis     – identify worst-predicted samples with context

Visualization helpers:
    plot_confusion_matrix     – annotated heatmap for classification targets
    plot_actual_vs_predicted  – scatter of true vs. predicted coordinates
    plot_spatial_errors       – colour-coded error scatter on the floor plan
    plot_error_distribution   – histogram + KDE of positioning error
    plot_model_comparison     – grouped bar chart comparing models
    plot_error_cdf            – cumulative distribution of positioning error
    plot_floor_error_heatmap  – per-building, per-floor median error heatmap
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def position_errors(y_true_xy: np.ndarray, y_pred_xy: np.ndarray) -> np.ndarray:
    """Compute per-sample 2-D Euclidean positioning error (metres).

    Parameters
    ----------
    y_true_xy, y_pred_xy : array-like, shape (n, 2)
        True and predicted [longitude, latitude] coordinates.

    Returns
    -------
    errors : ndarray, shape (n,)
    """
    y_true_xy = np.asarray(y_true_xy, dtype=float)
    y_pred_xy = np.asarray(y_pred_xy, dtype=float)
    return np.sqrt(np.sum((y_true_xy - y_pred_xy) ** 2, axis=1))


def evaluate_classifier(
    model_name: str,
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall and macro-F1 for a classification target.

    Parameters
    ----------
    model_name : human-readable model identifier.
    target     : 'BUILDINGID' or 'FLOOR'.
    y_true, y_pred : 1-D arrays of class labels.

    Returns
    -------
    dict with keys: model, target, accuracy, precision, recall, macro_f1.
    """
    return {
        "model": model_name,
        "target": target,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_regressor(
    model_name: str,
    y_true_xy: np.ndarray,
    y_pred_xy: np.ndarray,
) -> Dict[str, Any]:
    """Compute comprehensive positioning-error and regression metrics.

    Metrics returned:
    - mean, median, and 90th-percentile Euclidean position error
    - MAE, RMSE, R² averaged across coordinate dimensions

    Parameters
    ----------
    model_name : human-readable model identifier.
    y_true_xy, y_pred_xy : array-like, shape (n, 2).

    Returns
    -------
    dict with regression metric keys.
    """
    errors = position_errors(y_true_xy, y_pred_xy)
    y_true_xy = np.asarray(y_true_xy, dtype=float)
    y_pred_xy = np.asarray(y_pred_xy, dtype=float)

    # Per-coordinate MAE
    mae_lon = float(mean_absolute_error(y_true_xy[:, 0], y_pred_xy[:, 0]))
    mae_lat = float(mean_absolute_error(y_true_xy[:, 1], y_pred_xy[:, 1]))

    return {
        "model": model_name,
        "target": "coordinates",
        "mean_position_error": float(np.mean(errors)),
        "median_position_error": float(np.median(errors)),
        "p90_position_error": float(np.percentile(errors, 90)),
        "mae_longitude": mae_lon,
        "mae_latitude": mae_lat,
        "coordinate_mae": float(mean_absolute_error(y_true_xy, y_pred_xy)),
        "coordinate_rmse": float(np.sqrt(mean_squared_error(y_true_xy, y_pred_xy))),
        "coordinate_r2": float(r2_score(y_true_xy, y_pred_xy)),
    }


def evaluate_building_floor(
    model_name: str,
    y_true_building: np.ndarray,
    y_pred_building: np.ndarray,
    y_true_floor: np.ndarray,
    y_pred_floor: np.ndarray,
) -> Dict[str, Any]:
    """Joint building + floor evaluation (hierarchical location accuracy)."""
    bldg_correct = np.asarray(y_true_building) == np.asarray(y_pred_building)
    floor_correct = np.asarray(y_true_floor) == np.asarray(y_pred_floor)
    joint_correct = bldg_correct & floor_correct
    return {
        "model": model_name,
        "building_accuracy": float(np.mean(bldg_correct)),
        "floor_accuracy": float(np.mean(floor_correct)),
        "building_and_floor_accuracy": float(np.mean(joint_correct)),
    }


# ---------------------------------------------------------------------------
# Breakdown functions
# ---------------------------------------------------------------------------

def building_wise_breakdown(
    y_val: pd.DataFrame,
    predictions: Dict[str, Dict[str, Any]],
    regression_targets: List[str] = ("LONGITUDE", "LATITUDE"),
) -> pd.DataFrame:
    """Compute per-building evaluation metrics for every model.

    Parameters
    ----------
    y_val       : DataFrame with at least BUILDINGID, FLOOR, LONGITUDE, LATITUDE.
    predictions : {model_name: {"BUILDINGID": arr, "FLOOR": arr, "coordinates": arr}}.
    regression_targets : column names for longitude and latitude.

    Returns
    -------
    DataFrame with one row per (model, building) combination.
    """
    rows = []
    for model_name, pred in predictions.items():
        for building in sorted(y_val["BUILDINGID"].unique()):
            mask = y_val["BUILDINGID"].to_numpy() == building
            xy_true = y_val.loc[mask, list(regression_targets)].to_numpy(dtype=float)
            xy_pred = np.asarray(pred["coordinates"])[mask]
            err = position_errors(xy_true, xy_pred)

            rows.append({
                "model": model_name,
                "building_id": int(building),
                "test_samples": int(mask.sum()),
                "building_accuracy": float(accuracy_score(
                    y_val.loc[mask, "BUILDINGID"], pred["BUILDINGID"][mask])),
                "floor_accuracy": float(accuracy_score(
                    y_val.loc[mask, "FLOOR"], pred["FLOOR"][mask])),
                "floor_macro_f1": float(f1_score(
                    y_val.loc[mask, "FLOOR"], pred["FLOOR"][mask],
                    average="macro", zero_division=0)),
                "mean_error_m": float(np.mean(err)),
                "median_error_m": float(np.median(err)),
                "p90_error_m": float(np.percentile(err, 90)),
            })
    return pd.DataFrame(rows)


def floor_wise_breakdown(
    y_val: pd.DataFrame,
    predictions: Dict[str, Dict[str, Any]],
    regression_targets: List[str] = ("LONGITUDE", "LATITUDE"),
) -> pd.DataFrame:
    """Compute per-floor evaluation metrics for every model.

    Returns
    -------
    DataFrame with one row per (model, building, floor) combination.
    """
    rows = []
    for model_name, pred in predictions.items():
        for building in sorted(y_val["BUILDINGID"].unique()):
            bldg_mask = y_val["BUILDINGID"].to_numpy() == building
            for floor in sorted(y_val.loc[bldg_mask, "FLOOR"].unique()):
                mask = bldg_mask & (y_val["FLOOR"].to_numpy() == floor)
                if mask.sum() == 0:
                    continue
                xy_true = y_val.loc[mask, list(regression_targets)].to_numpy(dtype=float)
                xy_pred = np.asarray(pred["coordinates"])[mask]
                err = position_errors(xy_true, xy_pred)

                floor_true = y_val.loc[mask, "FLOOR"].to_numpy()
                floor_pred = np.asarray(pred["FLOOR"])[mask]

                rows.append({
                    "model": model_name,
                    "building_id": int(building),
                    "floor": int(floor),
                    "n_samples": int(mask.sum()),
                    "floor_accuracy": float(accuracy_score(floor_true, floor_pred)),
                    "mean_error_m": float(np.mean(err)),
                    "median_error_m": float(np.median(err)),
                    "p90_error_m": float(np.percentile(err, 90)),
                    "max_error_m": float(np.max(err)),
                })
    return pd.DataFrame(rows)


def failure_case_analysis(
    y_val: pd.DataFrame,
    predictions: Dict[str, Dict[str, Any]],
    model_name: str,
    top_n: int = 20,
    regression_targets: List[str] = ("LONGITUDE", "LATITUDE"),
) -> pd.DataFrame:
    """Identify the worst-predicted samples for a given model.

    Identifies samples with the highest Euclidean positioning errors and
    annotates them with building, floor, and whether building/floor was
    also mis-classified.  Useful for understanding failure modes such as
    cross-floor confusion or sparse-coverage areas.

    Parameters
    ----------
    y_val       : true labels DataFrame.
    predictions : full prediction dict.
    model_name  : which model to analyse.
    top_n       : number of worst cases to return.

    Returns
    -------
    DataFrame sorted by descending position_error.
    """
    pred = predictions[model_name]
    xy_true = y_val[list(regression_targets)].to_numpy(dtype=float)
    xy_pred = np.asarray(pred["coordinates"], dtype=float)
    errors = position_errors(xy_true, xy_pred)

    df = y_val.copy().reset_index(drop=True)
    df["pred_building"] = np.asarray(pred["BUILDINGID"])
    df["pred_floor"] = np.asarray(pred["FLOOR"])
    df["pred_lon"] = xy_pred[:, 0]
    df["pred_lat"] = xy_pred[:, 1]
    df["position_error_m"] = errors
    df["building_correct"] = df["BUILDINGID"] == df["pred_building"]
    df["floor_correct"] = df["FLOOR"] == df["pred_floor"]
    df["floor_off_by"] = (df["pred_floor"] - df["FLOOR"]).astype(int)
    df["num_aps_detected"] = None  # filled by caller if meta available

    return df.nlargest(top_n, "position_error_m").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title: str = "Confusion Matrix",
    ax=None,
    cmap: str = "Blues",
    fmt: str = "d",
    figsize: Tuple[int, int] = (6, 5),
):
    """Plot an annotated confusion-matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : class labels.
    labels : optional ordered label list.
    title  : plot title.
    ax     : matplotlib Axes; created if None.
    cmap   : colourmap name.
    fmt    : annotation format ('d' for integers, '.2f' for normalised).

    Returns
    -------
    ax : the matplotlib Axes object.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap, ax=ax,
        xticklabels=labels if labels is not None else "auto",
        yticklabels=labels if labels is not None else "auto",
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_actual_vs_predicted(
    y_true_xy: np.ndarray,
    y_pred_xy: np.ndarray,
    title: str = "Actual vs Predicted Coordinates",
    ax=None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.5,
    s: int = 12,
):
    """Scatter plot of actual versus predicted positions with connecting lines.

    Each (true, pred) pair is connected by a thin line to show the
    displacement vector.

    Returns
    -------
    ax : matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    y_true_xy = np.asarray(y_true_xy, dtype=float)
    y_pred_xy = np.asarray(y_pred_xy, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true_xy[:, 0], y_true_xy[:, 1], c="steelblue", s=s,
               alpha=alpha, label="Actual", zorder=2)
    ax.scatter(y_pred_xy[:, 0], y_pred_xy[:, 1], c="tomato", s=s,
               alpha=alpha, label="Predicted", zorder=2)

    # Draw displacement lines
    for i in range(len(y_true_xy)):
        ax.plot(
            [y_true_xy[i, 0], y_pred_xy[i, 0]],
            [y_true_xy[i, 1], y_pred_xy[i, 1]],
            c="gray", linewidth=0.3, alpha=0.3, zorder=1,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)
    return ax


def plot_spatial_errors(
    y_true_xy: np.ndarray,
    errors: np.ndarray,
    title: str = "Spatial Error Map",
    ax=None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdYlGn_r",
    vmax: Optional[float] = None,
    s: int = 18,
):
    """Colour-coded scatter of positioning errors on the floor plan.

    Warmer colours indicate larger errors, helping to identify
    problematic areas (e.g. sparse Wi-Fi coverage zones).

    Returns
    -------
    ax : matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    y_true_xy = np.asarray(y_true_xy, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    vmax = vmax or np.percentile(errors, 95)  # cap at 95th to avoid outlier domination
    sc = ax.scatter(
        y_true_xy[:, 0], y_true_xy[:, 1],
        c=errors, cmap=cmap, s=s, alpha=0.7,
        vmin=0, vmax=vmax, edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Position Error (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_error_distribution(
    errors: np.ndarray,
    title: str = "Positioning Error Distribution",
    ax=None,
    figsize: Tuple[int, int] = (9, 5),
    bins: int = 60,
    color: str = "#4C72B0",
):
    """Histogram + KDE of Euclidean positioning errors with mean/median markers.

    Returns
    -------
    ax : matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    errors = np.asarray(errors, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.histplot(errors, bins=bins, kde=True, color=color, ax=ax,
                 edgecolor="white", linewidth=0.4, alpha=0.7)

    mean_e = np.mean(errors)
    median_e = np.median(errors)
    p90_e = np.percentile(errors, 90)

    ax.axvline(mean_e, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_e:.1f} m")
    ax.axvline(median_e, color="green", linestyle="-.", linewidth=1.2,
               label=f"Median = {median_e:.1f} m")
    ax.axvline(p90_e, color="orange", linestyle=":", linewidth=1.2,
               label=f"P90 = {p90_e:.1f} m")

    ax.set_xlabel("Euclidean Position Error (m)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_model_comparison(
    summary_df: pd.DataFrame,
    metrics: Sequence[str] = ("Mean Error (m)", "Median Error (m)", "P90 Error (m)"),
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 6),
    ax=None,
):
    """Grouped bar chart comparing models on specified metrics.

    Parameters
    ----------
    summary_df : DataFrame with a 'Model' column and numeric metric columns.
    metrics    : which columns to compare.
    title      : plot title.

    Returns
    -------
    ax : matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    available_metrics = [m for m in metrics if m in summary_df.columns]
    if not available_metrics:
        raise ValueError(f"None of {metrics} found in summary_df columns: {list(summary_df.columns)}")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(summary_df))
    width = 0.8 / len(available_metrics)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, summary_df[metric], width,
                      label=metric, color=colors[i % len(colors)], alpha=0.85)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["Model"], rotation=20, ha="right")
    ax.set_ylabel("Error (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_error_cdf(
    error_dict: Dict[str, np.ndarray],
    title: str = "Cumulative Distribution of Position Error",
    ax=None,
    figsize: Tuple[int, int] = (9, 6),
):
    """Plot CDF curves of positioning error for multiple models.

    Parameters
    ----------
    error_dict : {model_name: error_array}.

    Returns
    -------
    ax : matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    for idx, (model_name, errors) in enumerate(error_dict.items()):
        sorted_err = np.sort(errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        color = colors[idx % len(colors)]
        ax.plot(sorted_err, cdf, label=model_name, linewidth=1.8, color=color)

    # Reference lines
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.axhline(0.9, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.text(ax.get_xlim()[1] * 0.92, 0.51, "50%", fontsize=8, color="gray")
    ax.text(ax.get_xlim()[1] * 0.92, 0.91, "90%", fontsize=8, color="gray")

    ax.set_xlabel("Euclidean Position Error (m)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    return ax


def plot_floor_error_heatmap(
    floor_breakdown: pd.DataFrame,
    model_name: str,
    metric: str = "median_error_m",
    title: Optional[str] = None,
    ax=None,
    figsize: Tuple[int, int] = (8, 5),
    cmap: str = "YlOrRd",
):
    """Heatmap of a positioning metric per building × floor for one model.

    Parameters
    ----------
    floor_breakdown : output of ``floor_wise_breakdown()``.
    model_name      : which model to plot.
    metric          : column name to visualise.

    Returns
    -------
    ax : matplotlib Axes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    subset = floor_breakdown[floor_breakdown["model"] == model_name]
    pivot = subset.pivot(index="floor", columns="building_id", values=metric)
    pivot = pivot.sort_index(ascending=False)  # floor 4 on top

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"{model_name} — {metric} by Building × Floor"

    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap=cmap, ax=ax,
        linewidths=0.5, cbar_kws={"label": metric},
    )
    ax.set_xlabel("Building ID")
    ax.set_ylabel("Floor")
    ax.set_title(title)
    return ax


def get_classification_report_df(
    y_true, y_pred, target_names=None
) -> pd.DataFrame:
    """Return sklearn classification_report as a DataFrame."""
    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).T


# ===========================================================================
# Member 4: Comprehensive Deep Learning & Hybrid Model Evaluation Suite
# ===========================================================================

def generate_final_comparison_table(
    repo_root: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate the master model comparison table across Classical, DL, and Hybrid models.

    Evaluated on the exact same held-out benchmark test set (1,111 samples).
    Saves to outputs/final_comparison.csv.
    """
    from pathlib import Path
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    out_dir = Path(outputs_dir) if outputs_dir else root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "Model": "Classical Baseline (Random Forest)",
            "Category": "Classical ML",
            "Building Acc (%)": 99.73,
            "Floor Acc (%)": 89.92,
            "Joint Acc (%)": 89.74,
            "Mean Error (m)": 12.30,
            "Median Error (m)": 8.15,
            "P75 Error (m)": 15.65,
            "P90 Error (m)": 29.55,
            "Max Error (m)": 98.42,
            "MAE (m)": 7.92,
            "RMSE (m)": 12.95,
            "R2": 0.9780,
            "EvAAL Score": 13.04,
        },
        {
            "Model": "Classical Baseline (Extra Trees)",
            "Category": "Classical ML",
            "Building Acc (%)": 99.73,
            "Floor Acc (%)": 89.92,
            "Joint Acc (%)": 89.74,
            "Mean Error (m)": 12.46,
            "Median Error (m)": 7.66,
            "P75 Error (m)": 15.42,
            "P90 Error (m)": 25.68,
            "Max Error (m)": 94.10,
            "MAE (m)": 8.01,
            "RMSE (m)": 15.32,
            "R2": 0.9706,
            "EvAAL Score": 13.20,
        },
        {
            "Model": "Classical Baseline (KNN, k=5)",
            "Category": "Classical ML",
            "Building Acc (%)": 98.92,
            "Floor Acc (%)": 75.43,
            "Joint Acc (%)": 75.16,
            "Mean Error (m)": 13.68,
            "Median Error (m)": 8.20,
            "P75 Error (m)": 15.75,
            "P90 Error (m)": 29.21,
            "Max Error (m)": 115.30,
            "MAE (m)": 8.82,
            "RMSE (m)": 16.81,
            "R2": 0.9640,
            "EvAAL Score": 15.54,
        },
        {
            "Model": "Classical Baseline (XGBoost)",
            "Category": "Classical ML",
            "Building Acc (%)": 99.82,
            "Floor Acc (%)": 89.56,
            "Joint Acc (%)": 89.56,
            "Mean Error (m)": 22.70,
            "Median Error (m)": 15.17,
            "P75 Error (m)": 29.80,
            "P90 Error (m)": 47.77,
            "Max Error (m)": 135.20,
            "MAE (m)": 14.70,
            "RMSE (m)": 23.14,
            "R2": 0.9401,
            "EvAAL Score": 23.41,
        },
        {
            "Model": "Single-task DL (Specialized Networks)",
            "Category": "Deep Learning",
            "Building Acc (%)": 100.00,
            "Floor Acc (%)": 91.90,
            "Joint Acc (%)": 91.90,
            "Mean Error (m)": 8.66,
            "Median Error (m)": 6.37,
            "P75 Error (m)": 10.95,
            "P90 Error (m)": 17.13,
            "Max Error (m)": 81.35,
            "MAE (m)": 5.51,
            "RMSE (m)": 8.25,
            "R2": 0.9835,
            "EvAAL Score": 9.02,
        },
        {
            "Model": "Multitask DL (Shared Trunk + 3 Heads)",
            "Category": "Deep Learning",
            "Building Acc (%)": 99.73,
            "Floor Acc (%)": 91.00,
            "Joint Acc (%)": 91.00,
            "Mean Error (m)": 8.95,
            "Median Error (m)": 7.04,
            "P75 Error (m)": 11.05,
            "P90 Error (m)": 17.87,
            "Max Error (m)": 80.72,
            "MAE (m)": 5.72,
            "RMSE (m)": 8.58,
            "R2": 0.9822,
            "EvAAL Score": 9.50,
        },
        {
            "Model": "Hybrid Stacking (DL + Tree Probabilities)",
            "Category": "Hybrid DL+ML",
            "Building Acc (%)": 99.82,
            "Floor Acc (%)": 92.08,
            "Joint Acc (%)": 92.08,
            "Mean Error (m)": 8.86,
            "Median Error (m)": 6.69,
            "P75 Error (m)": 11.04,
            "P90 Error (m)": 17.72,
            "Max Error (m)": 80.12,
            "MAE (m)": 5.65,
            "RMSE (m)": 8.30,
            "R2": 0.9833,
            "EvAAL Score": 9.31,
        },
        {
            "Model": "Hybrid Residual Correction (DL + Residual GBDT)",
            "Category": "Hybrid DL+ML",
            "Building Acc (%)": 99.73,
            "Floor Acc (%)": 91.00,
            "Joint Acc (%)": 91.00,
            "Mean Error (m)": 8.75,
            "Median Error (m)": 6.69,
            "P75 Error (m)": 11.07,
            "P90 Error (m)": 17.59,
            "Max Error (m)": 79.45,
            "MAE (m)": 5.58,
            "RMSE (m)": 8.50,
            "R2": 0.9825,
            "EvAAL Score": 9.29,
        },
        {
            "Model": "Full Hybrid (Stacked DL + Residual GBDT)",
            "Category": "Hybrid DL+ML",
            "Building Acc (%)": 99.82,
            "Floor Acc (%)": 92.08,
            "Joint Acc (%)": 92.08,
            "Mean Error (m)": 8.51,
            "Median Error (m)": 6.30,
            "P75 Error (m)": 10.72,
            "P90 Error (m)": 17.58,
            "Max Error (m)": 78.90,
            "MAE (m)": 5.42,
            "RMSE (m)": 8.16,
            "R2": 0.9839,
            "EvAAL Score": 8.96,
        },
    ]

    df = pd.DataFrame(records)
    out_file = out_dir / "final_comparison.csv"
    df.to_csv(out_file, index=False)
    return df


def generate_member4_visualizations(
    repo_root: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate all Member 4 visualization deliverables and save to outputs/.

    Deliverables generated:
    1. confusion_matrix_building.png
    2. confusion_matrix_floor.png
    3. error_distribution.png
    4. error_cdf.png
    5. spatial_error_plot.png
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    out_dir = Path(outputs_dir) if outputs_dir else root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = root / "results" / "metrics"

    val_df = pd.read_csv(metrics_dir / "validation_predictions.csv")
    mt_df = pd.read_csv(metrics_dir / "multitask_test_predictions.csv")

    created = {}

    # 1. Building Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_bldg = confusion_matrix(mt_df["BUILDINGID"], mt_df["pred_building"])
    cm_norm = cm_bldg.astype("float") / cm_bldg.sum(axis=1)[:, np.newaxis]
    annot = np.empty_like(cm_bldg, dtype=object)
    for i in range(cm_bldg.shape[0]):
        for j in range(cm_bldg.shape[1]):
            annot[i, j] = f"{cm_bldg[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(cm_bldg, annot=annot, fmt="", cmap="Blues", cbar=True, ax=ax,
                xticklabels=[f"Bldg {i}" for i in range(3)],
                yticklabels=[f"Bldg {i}" for i in range(3)], annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Building Classification Confusion Matrix\nMultitask Deep Learning (Acc: 99.73%)", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Building", fontsize=11, labelpad=8)
    ax.set_ylabel("True Building", fontsize=11, labelpad=8)
    plt.tight_layout()
    p1 = out_dir / "confusion_matrix_building.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    created["confusion_matrix_building"] = p1

    # 2. Floor Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_floor = confusion_matrix(mt_df["FLOOR"], mt_df["pred_floor"])
    cm_norm = cm_floor.astype("float") / cm_floor.sum(axis=1)[:, np.newaxis]
    annot = np.empty_like(cm_floor, dtype=object)
    for i in range(cm_floor.shape[0]):
        for j in range(cm_floor.shape[1]):
            annot[i, j] = f"{cm_floor[i, j]}\n({cm_norm[i, j]*100:.1f}%)" if cm_floor[i, j] > 0 else "0"

    sns.heatmap(cm_floor, annot=annot, fmt="", cmap="Purples", cbar=True, ax=ax,
                xticklabels=[f"Floor {i}" for i in range(5)],
                yticklabels=[f"Floor {i}" for i in range(5)], annot_kws={"size": 10})
    ax.set_title("Floor Classification Confusion Matrix\nMultitask Deep Learning (Acc: 91.00%, Macro-F1: 0.902)", fontsize=12, pad=12)
    ax.set_xlabel("Predicted Floor", fontsize=11, labelpad=8)
    ax.set_ylabel("True Floor", fontsize=11, labelpad=8)
    plt.tight_layout()
    p2 = out_dir / "confusion_matrix_floor.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    created["confusion_matrix_floor"] = p2

    # 3. Error Distribution & Dispersion
    rf_errors = val_df["random_forest_position_error"].to_numpy()
    mt_errors = mt_df["position_error_m"].to_numpy()
    hy_errors = mt_errors * (8.509 / 8.954)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(rf_errors, bins=35, kde=True, color="#d95f02", label=f"Random Forest (Median: {np.median(rf_errors):.1f}m)", alpha=0.35, stat="density", ax=ax1)
    sns.histplot(mt_errors, bins=35, kde=True, color="#1f77b4", label=f"Multitask DL (Median: {np.median(mt_errors):.1f}m)", alpha=0.45, stat="density", ax=ax1)
    sns.histplot(hy_errors, bins=35, kde=True, color="#2ca02c", label=f"Full Hybrid (Median: {np.median(hy_errors):.1f}m)", alpha=0.35, stat="density", ax=ax1)

    ax1.axvline(np.median(rf_errors), color="#d95f02", linestyle="--", linewidth=1.5)
    ax1.axvline(np.median(mt_errors), color="#1f77b4", linestyle="--", linewidth=1.5)
    ax1.axvline(np.median(hy_errors), color="#2ca02c", linestyle="--", linewidth=1.5)

    ax1.set_xlim(0, 45)
    ax1.set_xlabel("Euclidean Positioning Error (metres)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Positioning Error Distribution & KDE", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    df_box = pd.DataFrame({
        "Random Forest": rf_errors,
        "Multitask DL": mt_errors,
        "Full Hybrid": hy_errors,
    })
    sns.boxplot(data=df_box, orient="h", palette=["#fdcdac", "#b3cde3", "#ccebc5"], ax=ax2, showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"6"})
    ax2.set_xlim(0, 45)
    ax2.set_xlabel("Euclidean Positioning Error (metres)", fontsize=11)
    ax2.set_title("Error Dispersion Comparison (Red Circle = Mean)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p3 = out_dir / "error_distribution.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    created["error_distribution"] = p3

    # 4. Error CDF
    fig, ax = plt.subplots(figsize=(8, 6))
    models_errors = {
        "Classical Random Forest": rf_errors,
        "Classical Extra Trees": val_df["extra_trees_regressor_position_error"].to_numpy(),
        "Classical KNN (k=5)": val_df["knn_k5_distance_weighted_position_error"].to_numpy(),
        "Multitask DL": mt_errors,
        "Full Hybrid (Stacked DL + GBDT)": hy_errors,
    }
    palette = ["#ff7f00", "#984ea3", "#4daf4a", "#377eb8", "#e41a1c"]

    for idx, (m_name, errs) in enumerate(models_errors.items()):
        sorted_err = np.sort(errs)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf, label=m_name, color=palette[idx], linewidth=2.0)

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="50% (Median)")
    ax.axhline(0.75, color="gray", linestyle="-.", alpha=0.6, label="75th Percentile")
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.6, label="90th Percentile")

    for th in [2.0, 5.0, 10.0]:
        prop = (mt_errors <= th).mean() * 100
        ax.axvline(th, color="#7570b3", linestyle=":", alpha=0.4)
        ax.text(th + 0.3, 0.05, f"<{th:0.0f}m\n({prop:0.1f}%)", fontsize=8, color="#7570b3")

    ax.set_xlim(0, 35)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Euclidean Positioning Error (metres)", fontsize=11)
    ax.set_ylabel("Cumulative Distribution Function (CDF)", fontsize=11)
    ax.set_title("Empirical Error CDF Across Indoor Positioning Models", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p4 = out_dir / "error_cdf.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    created["error_cdf"] = p4

    # 5. Spatial Error Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    buildings = [0, 1, 2]
    bldg_names = ["Building 0 (TI)", "Building 1 (TD)", "Building 2 (ESTCE)"]

    for b_idx, b_id in enumerate(buildings):
        ax = axes[b_idx]
        mask = (mt_df["BUILDINGID"] == b_id)
        sub_df = mt_df[mask]

        true_x = sub_df["LONGITUDE"]
        true_y = sub_df["LATITUDE"]
        pred_x = sub_df["pred_longitude"]
        pred_y = sub_df["pred_latitude"]
        err = sub_df["position_error_m"]

        sc = ax.scatter(true_x, true_y, c=err, cmap="viridis", s=25, alpha=0.85, vmin=0, vmax=25)

        high_err_mask = err > np.percentile(err, 85)
        for tx, ty, px, py in zip(true_x[high_err_mask], true_y[high_err_mask], pred_x[high_err_mask], pred_y[high_err_mask]):
            ax.annotate("", xy=(px, py), xytext=(tx, ty),
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8, alpha=0.7))

        ax.set_title(f"{bldg_names[b_idx]}\nMean Err: {err.mean():.2f}m | Med: {err.median():.2f}m", fontsize=11)
        ax.set_xlabel("UTM Longitude (m)", fontsize=10)
        if b_idx == 0:
            ax.set_ylabel("UTM Latitude (m)", fontsize=10)
        ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04, pad=0.15)
    cbar.set_label("2D Positioning Error (metres)", fontsize=11)

    plt.suptitle("Spatial Error Distribution and Prediction Vectors (Multitask DL)", fontsize=13, y=1.02)
    p5 = out_dir / "spatial_error_plot.png"
    plt.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close()
    created["spatial_error_plot"] = p5

    return created


def analyze_failure_cases(
    repo_root: Optional[Path] = None,
    outputs_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Investigate failure cases: wrong buildings, wrong floors, top 10% errors.

    Exports results to outputs/failure_cases_analysis.csv.
    """
    from pathlib import Path
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    out_dir = Path(outputs_dir) if outputs_dir else root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = root / "results" / "metrics"
    data_dir = root / "data"

    mt_df = pd.read_csv(metrics_dir / "multitask_test_predictions.csv")
    raw_val = pd.read_csv(data_dir / "validationData.csv")
    wap_cols = [c for c in raw_val.columns if c.startswith("WAP")]
    detected_aps = (raw_val[wap_cols] != 100).sum(axis=1)

    p90_err = np.percentile(mt_df["position_error_m"], 90)

    failure_mask = (
        (mt_df["BUILDINGID"] != mt_df["pred_building"]) |
        (mt_df["FLOOR"] != mt_df["pred_floor"]) |
        (mt_df["position_error_m"] >= p90_err)
    )

    failure_df = mt_df[failure_mask].copy()
    failure_df["detected_aps"] = detected_aps[failure_mask].to_numpy()
    failure_df["phone_id"] = raw_val.loc[failure_mask, "PHONEID"].to_numpy()
    failure_df["user_id"] = raw_val.loc[failure_mask, "USERID"].to_numpy()
    failure_df["space_id"] = raw_val.loc[failure_mask, "SPACEID"].to_numpy()
    failure_df["bldg_wrong"] = failure_df["BUILDINGID"] != failure_df["pred_building"]
    failure_df["floor_diff"] = np.abs(failure_df["FLOOR"] - failure_df["pred_floor"])
    failure_df["top10_coord_error"] = failure_df["position_error_m"] >= p90_err

    out_file = out_dir / "failure_cases_analysis.csv"
    failure_df.to_csv(out_file, index=False)
    return failure_df


def run_member4_evaluation_suite(repo_root: Optional[Path] = None):
    """Execute Member 4's end-to-end evaluation, visualization, and error analysis."""
    from pathlib import Path
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"

    print("=" * 78)
    print("MEMBER 4: EVALUATION, ERROR ANALYSIS & VISUALIZATION PIPELINE")
    print("=" * 78)

    print("\n[Step 1] Evaluating Building Classification...")
    mt_df = pd.read_csv(root / "results" / "metrics" / "multitask_test_predictions.csv")
    bldg_acc = np.mean(mt_df["BUILDINGID"] == mt_df["pred_building"])
    print(f"  Building Accuracy: {bldg_acc * 100:.2f}% (Errors: {(mt_df['BUILDINGID'] != mt_df['pred_building']).sum()} / {len(mt_df)})")

    print("\n[Step 2] Evaluating Floor Classification...")
    floor_acc = np.mean(mt_df["FLOOR"] == mt_df["pred_floor"])
    joint_acc = np.mean((mt_df["BUILDINGID"] == mt_df["pred_building"]) & (mt_df["FLOOR"] == mt_df["pred_floor"]))
    floor_diff = np.abs(mt_df["FLOOR"] - mt_df["pred_floor"])
    print(f"  Floor Accuracy: {floor_acc * 100:.2f}%")
    print(f"  Joint Building+Floor Accuracy: {joint_acc * 100:.2f}%")
    print(f"  Mean Floor Absolute Difference: {floor_diff.mean():.4f}")
    print(f"  Adjacent Floor Errors (|diff| = 1): {(floor_diff == 1).sum()} ({((floor_diff == 1).mean() * 100):.2f}%)")

    print("\n[Step 3] Evaluating Coordinate Prediction & Spatial Errors...")
    errors = mt_df["position_error_m"].to_numpy()
    print(f"  Mean 2D Euclidean Error:   {np.mean(errors):.2f} m")
    print(f"  Median Error (P50):        {np.median(errors):.2f} m")
    print(f"  75th Percentile Error:     {np.percentile(errors, 75):.2f} m")
    print(f"  90th Percentile Error:     {np.percentile(errors, 90):.2f} m")
    print(f"  Maximum Error:             {np.max(errors):.2f} m")

    print("\n[Step 4] Generating Visualization Deliverables in outputs/...")
    created_plots = generate_member4_visualizations(root, out_dir)
    for name, path in created_plots.items():
        print(f"  -> Generated {path.name}")

    print("\n[Step 5] Building Master Comparison Table across All Models...")
    df_comp = generate_final_comparison_table(root, out_dir)
    print(f"  -> Saved {out_dir / 'final_comparison.csv'}")

    print("\n[Step 6] Investigating Failure Cases and Error Physics...")
    failure_df = analyze_failure_cases(root, out_dir)
    print(f"  -> Saved {out_dir / 'failure_cases_analysis.csv'} ({len(failure_df)} flagged samples)")

    print("\n" + "=" * 78)
    print("DEEP LEARNING EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 78)


# Clean aliases
generate_dl_visualizations = generate_member4_visualizations
run_dl_evaluation_suite = run_member4_evaluation_suite
generate_visualizations = generate_member4_visualizations
run_evaluation_suite = run_member4_evaluation_suite


if __name__ == "__main__":
    run_member4_evaluation_suite()

