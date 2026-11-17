"""Residual coordinate correction via Gradient Boosted Decision Trees (Member 2).

This module implements Hybrid Approach B:
Train the deep neural network to predict continuous 2D coordinates (x, y) in metres,
and then train gradient boosted decision trees to model and correct systematic residual errors:

    e_x = x_true - x_hat_DL
    e_y = y_true - y_hat_DL

    x_hat_hybrid = x_hat_DL + e_hat_x
    y_hat_hybrid = y_hat_DL + e_hat_y

Why Residual Boosting Works
---------------------------
Neural networks approximate smooth continuous manifolds over Euclidean coordinates, but Wi-Fi
propagation is piecewise non-linear due to internal wall partitions, multipath scattering,
and antenna gain discrepancies across devices. Tree boosting partitions the feature space into
local hyper-rectangles, learning local spatial biases that the global neural network trunk smoothed out.

Leakage Prevention
------------------
The residual model is fitted EXCLUSIVELY on training-set residuals. Under no circumstances are
validation or test ground-truth coordinates exposed to the booster.
"""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESIDUAL_PATH = REPO_ROOT / "models" / "residual_booster.joblib"


class ResidualCoordinateCorrector:
    """Gradient boosted residual model for systematic 2D spatial coordinate correction."""

    def __init__(
        self,
        max_iter: int = 120,
        max_depth: int = 6,
        learning_rate: float = 0.04,
        l2_regularization: float = 2.0,
        use_pred_coords_as_features: bool = True,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.use_pred_coords_as_features = use_pred_coords_as_features
        self.random_state = random_state

        self.booster_x: Optional[HistGradientBoostingRegressor] = None
        self.booster_y: Optional[HistGradientBoostingRegressor] = None

    def _prepare_features(self, X: np.ndarray, pred_coords_m: np.ndarray) -> np.ndarray:
        """Combine RSSI features with DL coordinate predictions if configured."""
        if self.use_pred_coords_as_features:
            return np.hstack([X, pred_coords_m]).astype(np.float32)
        return X.astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        true_coords_m_train: np.ndarray,
        dl_coords_m_train: np.ndarray,
    ) -> "ResidualCoordinateCorrector":
        """Compute training residuals and fit boosting regressors for delta-x and delta-y."""
        # Residual error = ground truth - neural estimate
        residuals_x = true_coords_m_train[:, 0] - dl_coords_m_train[:, 0]
        residuals_y = true_coords_m_train[:, 1] - dl_coords_m_train[:, 1]

        features_train = self._prepare_features(X_train, dl_coords_m_train)

        self.booster_x = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
        )
        self.booster_y = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state + 1,
        )

        self.booster_x.fit(features_train, residuals_x)
        self.booster_y.fit(features_train, residuals_y)

        return self

    def predict_residuals(
        self, X: np.ndarray, dl_coords_m: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate systematic residual offsets e_hat_x and e_hat_y."""
        if self.booster_x is None or self.booster_y is None:
            raise RuntimeError("ResidualCoordinateCorrector must be fitted before calling predict.")
        features = self._prepare_features(X, dl_coords_m)
        pred_res_x = self.booster_x.predict(features)
        pred_res_y = self.booster_y.predict(features)
        return pred_res_x, pred_res_y

    def correct_coordinates(
        self, X: np.ndarray, dl_coords_m: np.ndarray
    ) -> np.ndarray:
        """Produce final corrected coordinate predictions in metres:
        x_hybrid = x_DL + e_hat_x
        y_hybrid = y_DL + e_hat_y
        """
        e_hat_x, e_hat_y = self.predict_residuals(X, dl_coords_m)
        corrected = np.column_stack([
            dl_coords_m[:, 0] + e_hat_x,
            dl_coords_m[:, 1] + e_hat_y,
        ])
        return corrected.astype(np.float64)

    def save(self, filepath: Path = DEFAULT_RESIDUAL_PATH):
        """Save fitted residual model to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path = DEFAULT_RESIDUAL_PATH) -> "ResidualCoordinateCorrector":
        """Load fitted residual model from disk."""
        return joblib.load(filepath)
