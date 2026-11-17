"""Metrics module for indoor Wi-Fi positioning evaluation (Member 4).

This module defines standard evaluation metrics for hierarchical indoor localization:
1. Building classification: accuracy, macro precision, recall, macro-F1, confusion matrix.
2. Floor classification: accuracy, macro precision, recall, macro-F1, mean absolute floor difference.
3. Joint classification: joint building + floor accuracy (both correct simultaneously).
4. Coordinate regression: Euclidean positioning errors (mean, median, p75, p90, max),
   per-axis MAE/RMSE, coordinate MAE/RMSE, R-squared score.
5. EvAAL competition composite penalty metric:
   EvAAL_score = mean_2d_error + 50 * building_penalty + 4 * floor_error
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

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


def compute_euclidean_errors(
    y_true_coords: Union[np.ndarray, pd.DataFrame],
    y_pred_coords: Union[np.ndarray, pd.DataFrame],
) -> np.ndarray:
    """Compute per-sample 2D Euclidean distance positioning error (in metres).

    Parameters
    ----------
    y_true_coords : array-like of shape (N, 2)
        Ground-truth [longitude, latitude] or [x, y] coordinates.
    y_pred_coords : array-like of shape (N, 2)
        Predicted [longitude, latitude] or [x, y] coordinates.

    Returns
    -------
    np.ndarray of shape (N,)
        Euclidean error for each sample in metres.
    """
    y_true = np.asarray(y_true_coords, dtype=float)
    y_pred = np.asarray(y_pred_coords, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    diff = y_true - y_pred
    return np.sqrt(np.sum(diff ** 2, axis=1))


def evaluate_building_classification(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Compute standard classification metrics for building identification.

    Returns
    -------
    dict with keys:
        - building_acc
        - building_precision_macro
        - building_recall_macro
        - building_f1_macro
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "building_acc": float(accuracy_score(y_true, y_pred)),
        "building_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "building_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "building_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_floor_classification(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Compute standard classification metrics for floor identification.

    Returns
    -------
    dict with keys:
        - floor_acc
        - floor_precision_macro
        - floor_recall_macro
        - floor_f1_macro
        - floor_abs_diff (mean absolute difference between true and predicted floor)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_diff = float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))
    return {
        "floor_acc": float(accuracy_score(y_true, y_pred)),
        "floor_precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "floor_recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "floor_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "floor_abs_diff": abs_diff,
    }


def compute_joint_accuracy(
    y_true_bldg: Union[np.ndarray, pd.Series],
    y_pred_bldg: Union[np.ndarray, pd.Series],
    y_true_floor: Union[np.ndarray, pd.Series],
    y_pred_floor: Union[np.ndarray, pd.Series],
) -> float:
    """Compute joint building + floor accuracy.

    A prediction is correct only if BOTH building and floor match the ground truth.
    """
    y_true_bldg = np.asarray(y_true_bldg)
    y_pred_bldg = np.asarray(y_pred_bldg)
    y_true_floor = np.asarray(y_true_floor)
    y_pred_floor = np.asarray(y_pred_floor)

    correct = (y_true_bldg == y_pred_bldg) & (y_true_floor == y_pred_floor)
    return float(np.mean(correct))


def compute_positioning_metrics(
    errors_m: np.ndarray,
    y_true_coords: Optional[np.ndarray] = None,
    y_pred_coords: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute comprehensive statistical metrics from 2D Euclidean errors.

    Parameters
    ----------
    errors_m : 1D array of Euclidean errors in metres
    y_true_coords, y_pred_coords : optional (N, 2) arrays for MAE, RMSE, R2 computation

    Returns
    -------
    dict of metrics:
        - mean_error_m
        - median_error_m
        - p75_error_m
        - p90_error_m
        - max_error_m
        - coord_mae_m
        - coord_rmse_m
        - r2_score (if coords provided)
    """
    errors = np.asarray(errors_m, dtype=float)
    metrics: Dict[str, float] = {
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "p75_error_m": float(np.percentile(errors, 75)),
        "p90_error_m": float(np.percentile(errors, 90)),
        "max_error_m": float(np.max(errors)),
    }

    if y_true_coords is not None and y_pred_coords is not None:
        y_true = np.asarray(y_true_coords, dtype=float)
        y_pred = np.asarray(y_pred_coords, dtype=float)
        metrics["mae_longitude_m"] = float(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))
        metrics["mae_latitude_m"] = float(mean_absolute_error(y_true[:, 1], y_pred[:, 1]))
        metrics["coord_mae_m"] = float(mean_absolute_error(y_true, y_pred))
        metrics["coord_rmse_m"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["r2_score"] = float(r2_score(y_true, y_pred))

    return metrics


def compute_evaal_score(
    errors_m: np.ndarray,
    y_true_bldg: np.ndarray,
    y_pred_bldg: np.ndarray,
    y_true_floor: np.ndarray,
    y_pred_floor: np.ndarray,
    building_penalty: float = 50.0,
    floor_penalty: float = 4.0,
) -> float:
    """Compute the international EvAAL (Evaluating Ambient Assisted Living) competition metric.

    EvAAL score penalizes incorrect building and floor predictions:
        Sample Penalty = 2D_error + building_penalty * (bldg_wrong) + floor_penalty * |floor_true - floor_pred|
    EvAAL Score = Mean across all samples.
    Lower score is better.
    """
    errors = np.asarray(errors_m, dtype=float)
    y_true_b = np.asarray(y_true_bldg)
    y_pred_b = np.asarray(y_pred_bldg)
    y_true_f = np.asarray(y_true_floor, dtype=float)
    y_pred_f = np.asarray(y_pred_floor, dtype=float)

    bldg_wrong = (y_true_b != y_pred_b).astype(float)
    floor_diff = np.abs(y_true_f - y_pred_f)

    penalties = errors + (building_penalty * bldg_wrong) + (floor_penalty * floor_diff)
    return float(np.mean(penalties))


def evaluate_full_system(
    y_true_bldg: np.ndarray,
    y_pred_bldg: np.ndarray,
    y_true_floor: np.ndarray,
    y_pred_floor: np.ndarray,
    y_true_coords: np.ndarray,
    y_pred_coords: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Calculate the complete evaluation suite across all targets in one call.

    Returns
    -------
    dict with unified keys matching the project's comparison tables.
    """
    bldg_metrics = evaluate_building_classification(y_true_bldg, y_pred_bldg)
    floor_metrics = evaluate_floor_classification(y_true_floor, y_pred_floor)
    joint_acc = compute_joint_accuracy(y_true_bldg, y_pred_bldg, y_true_floor, y_pred_floor)
    errors = compute_euclidean_errors(y_true_coords, y_pred_coords)
    coord_metrics = compute_positioning_metrics(errors, y_true_coords, y_pred_coords)
    evaal = compute_evaal_score(errors, y_true_bldg, y_pred_bldg, y_true_floor, y_pred_floor)

    return {
        "Model": model_name,
        "Building Acc": bldg_metrics["building_acc"],
        "Building F1": bldg_metrics["building_f1_macro"],
        "Floor Acc": floor_metrics["floor_acc"],
        "Floor F1": floor_metrics["floor_f1_macro"],
        "Floor Abs Diff": floor_metrics["floor_abs_diff"],
        "Joint Acc": joint_acc,
        "Mean Error (m)": coord_metrics["mean_error_m"],
        "Median Error (m)": coord_metrics["median_error_m"],
        "P75 Error (m)": coord_metrics["p75_error_m"],
        "P90 Error (m)": coord_metrics["p90_error_m"],
        "Max Error (m)": coord_metrics["max_error_m"],
        "MAE (m)": coord_metrics.get("coord_mae_m", np.nan),
        "RMSE (m)": coord_metrics.get("coord_rmse_m", np.nan),
        "R2": coord_metrics.get("r2_score", np.nan),
        "EvAAL Score": evaal,
    }
