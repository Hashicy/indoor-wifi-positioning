"""Unit tests for the Hybrid Deep Learning + Machine Learning components (Member 2).

Run with:
    .conda/bin/python -m pytest tests/test_hybrid.py
"""

import numpy as np
import pytest
import torch

from src.models.hybrid_model import HybridPositioningSystem
from src.models.multitask_net import ALL_TASKS, MultitaskNet
from src.residual_correction import ResidualCoordinateCorrector
from src.stacking_features import TreeProbabilityExtractor, concatenate_hybrid_features


def synthetic_data(n_samples=100, n_features=20, n_buildings=3, n_floors=5, seed=42):
    """Generate minimal synthetic data for fast unit testing."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    building = rng.integers(0, n_buildings, size=n_samples)
    floor = rng.integers(0, n_floors, size=n_samples)
    coords_m = rng.uniform(-100.0, 100.0, size=(n_samples, 2)).astype(np.float64)
    return X, building, floor, coords_m


class DummyScaler:
    """Mock standard scaler for coordinate inversion."""
    def inverse_transform(self, X):
        return X * 10.0 + 50.0


class DummyEncoder:
    """Mock label encoder for building/floor inversion."""
    def __init__(self, n_classes):
        self.classes_ = np.arange(n_classes)

    def inverse_transform(self, indices):
        return np.array([self.classes_[i] for i in indices])


def test_tree_probability_extractor_shapes_and_probabilities():
    X, bld, flr, _ = synthetic_data(n_samples=60, n_features=15)
    extractor = TreeProbabilityExtractor(n_estimators=10, n_splits=3, random_state=42)

    oof_bld, oof_flr = extractor.fit_transform_train(X, bld, flr)

    # Output shapes
    assert oof_bld.shape == (60, 3)
    assert oof_flr.shape == (60, 5)

    # Probabilities must sum to 1.0 per sample
    assert np.allclose(oof_bld.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(oof_flr.sum(axis=1), 1.0, atol=1e-5)

    # Test transform on unseen samples
    X_test, _, _, _ = synthetic_data(n_samples=20, n_features=15, seed=99)
    test_bld_p, test_flr_p = extractor.transform(X_test)
    assert test_bld_p.shape == (20, 3)
    assert test_flr_p.shape == (20, 5)
    assert np.allclose(test_bld_p.sum(axis=1), 1.0, atol=1e-5)


def test_concatenate_hybrid_features():
    X, _, _, _ = synthetic_data(n_samples=25, n_features=10)
    p_bld = np.ones((25, 3), dtype=np.float32) / 3.0
    p_flr = np.ones((25, 5), dtype=np.float32) / 5.0

    hybrid_X = concatenate_hybrid_features(X, p_bld, p_flr)
    assert hybrid_X.shape == (25, 10 + 3 + 5)
    assert np.allclose(hybrid_X[:, 10:13], p_bld)
    assert np.allclose(hybrid_X[:, 13:18], p_flr)


def test_residual_coordinate_corrector():
    X, _, _, true_coords = synthetic_data(n_samples=80, n_features=10)
    # Simulate a neural network with a constant + random bias
    dl_coords = true_coords + np.array([2.5, -4.0])

    corrector = ResidualCoordinateCorrector(max_iter=30, max_depth=3, random_state=42)
    corrector.fit(X, true_coords, dl_coords)

    # Predict residuals
    e_x, e_y = corrector.predict_residuals(X, dl_coords)
    assert e_x.shape == (80,)
    assert e_y.shape == (80,)

    # Correct coordinates
    corrected = corrector.correct_coordinates(X, dl_coords)
    assert corrected.shape == (80, 2)

    # Residual boosting should reduce the synthetic systematic bias on the training set
    initial_err = np.mean(np.linalg.norm(true_coords - dl_coords, axis=1))
    corrected_err = np.mean(np.linalg.norm(true_coords - corrected, axis=1))
    assert corrected_err < initial_err


def test_hybrid_positioning_system_modes_and_inference(tmp_path):
    X, bld, flr, true_coords = synthetic_data(n_samples=50, n_features=12)

    # Setup dummy tree extractor
    extractor = TreeProbabilityExtractor(n_estimators=10, n_splits=2, random_state=42)
    oof_bld, oof_flr = extractor.fit_transform_train(X, bld, flr)
    X_stacked = concatenate_hybrid_features(X, oof_bld, oof_flr)

    # Setup DL model
    dl_model = MultitaskNet(n_inputs=12 + 3 + 5, hidden=(32, 16, 8), dropout=0.0)

    # Setup Residual Corrector
    corrector = ResidualCoordinateCorrector(max_iter=20, max_depth=2, random_state=42)
    dummy_dl_pred = np.zeros((50, 2))
    corrector.fit(X_stacked, true_coords, dummy_dl_pred)

    prep = {
        "building_encoder": DummyEncoder(3),
        "floor_encoder": DummyEncoder(5),
        "coord_scaler": DummyScaler(),
    }

    # Test full hybrid mode
    system = HybridPositioningSystem(
        mode="full_hybrid",
        dl_model=dl_model,
        tree_extractor=extractor,
        residual_corrector=corrector,
    )

    preds = system.predict(X, prep, device=torch.device("cpu"))
    assert "building" in preds
    assert "floor" in preds
    assert "coords_m" in preds
    assert len(preds["building"]) == 50
    assert len(preds["floor"]) == 50
    assert preds["coords_m"].shape == (50, 2)

    # Test saving and loading
    save_file = tmp_path / "test_hybrid.joblib"
    system.save(save_file)
    assert save_file.exists()

    loaded_system = HybridPositioningSystem.load(save_file, device=torch.device("cpu"))
    loaded_preds = loaded_system.predict(X, prep, device=torch.device("cpu"))
    assert np.allclose(preds["coords_m"], loaded_preds["coords_m"], atol=1e-5)
