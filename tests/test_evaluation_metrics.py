"""Unit tests for Member 4 evaluation metrics (src/metrics.py).

Run with:
    /opt/anaconda3/bin/python -m pytest tests/test_evaluation_metrics.py
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


def test_compute_euclidean_errors():
    true_xy = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 20.0]])
    pred_xy = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 20.0]])
    errors = compute_euclidean_errors(true_xy, pred_xy)

    assert errors.shape == (3,)
    assert np.isclose(errors[0], 0.0)
    assert np.isclose(errors[1], 5.0)  # 3-4-5 triangle
    assert np.isclose(errors[2], 0.0)


def test_evaluate_building_classification():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 0])  # last one wrong

    res = evaluate_building_classification(y_true, y_pred)
    assert np.isclose(res["building_acc"], 5 / 6)
    assert "building_precision_macro" in res
    assert "building_recall_macro" in res
    assert "building_f1_macro" in res


def test_evaluate_floor_classification():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 3, 3, 4])  # sample 2 predicted as 3 (diff = 1)

    res = evaluate_floor_classification(y_true, y_pred)
    assert np.isclose(res["floor_acc"], 4 / 5)
    assert np.isclose(res["floor_abs_diff"], 1 / 5)
    assert "floor_f1_macro" in res


def test_compute_joint_accuracy():
    y_true_bldg = np.array([0, 1, 2, 0])
    y_pred_bldg = np.array([0, 1, 2, 1])  # sample 3 wrong bldg

    y_true_floor = np.array([0, 1, 2, 0])
    y_pred_floor = np.array([0, 2, 2, 0])  # sample 1 wrong floor

    # sample 0: correct, sample 1: wrong floor, sample 2: correct, sample 3: wrong bldg
    joint_acc = compute_joint_accuracy(y_true_bldg, y_pred_bldg, y_true_floor, y_pred_floor)
    assert np.isclose(joint_acc, 2 / 4)


def test_compute_positioning_metrics():
    errors = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    true_xy = np.zeros((5, 2))
    pred_xy = np.column_stack([errors, np.zeros(5)])

    metrics = compute_positioning_metrics(errors, true_xy, pred_xy)
    assert np.isclose(metrics["mean_error_m"], 6.0)
    assert np.isclose(metrics["median_error_m"], 6.0)
    assert np.isclose(metrics["max_error_m"], 10.0)
    assert "p75_error_m" in metrics
    assert "p90_error_m" in metrics
    assert "coord_mae_m" in metrics
    assert "coord_rmse_m" in metrics
    assert "r2_score" in metrics


def test_compute_evaal_score():
    errors = np.array([10.0, 5.0])
    y_true_bldg = np.array([0, 1])
    y_pred_bldg = np.array([1, 1])  # sample 0 wrong bldg -> penalty 50
    y_true_floor = np.array([0, 2])
    y_pred_floor = np.array([0, 1])  # sample 1 off by 1 floor -> penalty 4

    # Sample 0 penalty: 10 + 50*1 + 4*0 = 60
    # Sample 1 penalty: 5 + 50*0 + 4*1 = 9
    # EvAAL mean = (60 + 9) / 2 = 34.5
    evaal = compute_evaal_score(errors, y_true_bldg, y_pred_bldg, y_true_floor, y_pred_floor)
    assert np.isclose(evaal, 34.5)


def test_evaluate_full_system():
    n = 20
    rng = np.random.default_rng(42)
    bldg_true = rng.integers(0, 3, n)
    bldg_pred = bldg_true.copy()
    bldg_pred[0] = (bldg_pred[0] + 1) % 3

    floor_true = rng.integers(0, 5, n)
    floor_pred = floor_true.copy()
    floor_pred[1] = (floor_pred[1] + 1) % 5

    coords_true = rng.uniform(-100, 100, (n, 2))
    coords_pred = coords_true + rng.normal(0, 2, (n, 2))

    summary = evaluate_full_system(
        bldg_true, bldg_pred, floor_true, floor_pred, coords_true, coords_pred, model_name="TestModel"
    )

    assert summary["Model"] == "TestModel"
    assert "Building Acc" in summary
    assert "Floor Acc" in summary
    assert "Joint Acc" in summary
    assert "Mean Error (m)" in summary
    assert "Median Error (m)" in summary
    assert "P90 Error (m)" in summary
    assert "EvAAL Score" in summary
