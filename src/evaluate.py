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
