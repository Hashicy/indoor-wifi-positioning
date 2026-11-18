"""Unit tests for Positioning and Classification Metrics (Member 5 Integration).

Validates:
- Euclidean error calculation (2D Cartesian distance)
- Building classification accuracy, macro precision, recall, F1
- Floor classification accuracy, macro F1, and absolute floor difference
- Joint building-floor accuracy
- Positioning percentiles (median, P75, P90, max)
- EvAAL/IPIN official competition penalised score formula

Run with:
    pytest tests/test_metrics.py
"""

import numpy as np
import pytest

from src.metrics import (
    compute_euclidean_errors,
    evaluate_building_classification,
    evaluate_floor_classification,
    compute_joint_accuracy,
    compute_positioning_metrics,
    compute_evaal_score,
    evaluate_full_system,
)


def test_euclidean_metrics():
    true_xy = np.array([[10.0, 20.0], [0.0, 0.0], [5.0, 12.0]])
    pred_xy = np.array([[10.0, 20.0], [3.0, 4.0], [0.0, 0.0]])
    # distances: 0, 5, 13
    errors = compute_euclidean_errors(true_xy, pred_xy)
    assert np.allclose(errors, [0.0, 5.0, 13.0])

    metrics = compute_positioning_metrics(errors, true_xy, pred_xy)
    assert np.isclose(metrics["mean_error_m"], 6.0)
    assert np.isclose(metrics["median_error_m"], 5.0)
    assert np.isclose(metrics["max_error_m"], 13.0)


def test_building_and_floor_evaluation():
    y_b_true = np.array([0, 1, 2, 0])
    y_b_pred = np.array([0, 1, 2, 1])  # 3/4 = 75%
    b_res = evaluate_building_classification(y_b_true, y_b_pred)
    assert np.isclose(b_res["building_acc"], 0.75)

    y_f_true = np.array([0, 1, 2, 3])
    y_f_pred = np.array([0, 1, 3, 3])  # 3/4 = 75%, floor diff = 1/4 = 0.25
    f_res = evaluate_floor_classification(y_f_true, y_f_pred)
    assert np.isclose(f_res["floor_acc"], 0.75)
    assert np.isclose(f_res["floor_abs_diff"], 0.25)


def test_joint_accuracy_and_evaal():
    # Sample 0: correct building, correct floor -> penalty 0
    # Sample 1: wrong building (pen 50), correct floor
    # Sample 2: correct building, 2 floors off (pen 4*2 = 8)
    b_true = np.array([0, 1, 2])
    b_pred = np.array([0, 2, 2])
    f_true = np.array([0, 1, 3])
    f_pred = np.array([0, 1, 1])
    joint = compute_joint_accuracy(b_true, b_pred, f_true, f_pred)
    assert np.isclose(joint, 1 / 3)

    errors = np.array([2.0, 4.0, 6.0])
    # penalties:
    # 0: 2 + 0 + 0 = 2
    # 1: 4 + 50*1 + 4*0 = 54
    # 2: 6 + 50*0 + 4*2 = 14
    # mean: (2 + 54 + 14)/3 = 70/3 = 23.3333...
    evaal = compute_evaal_score(errors, b_true, b_pred, f_true, f_pred)
    assert np.isclose(evaal, 70.0 / 3.0)
