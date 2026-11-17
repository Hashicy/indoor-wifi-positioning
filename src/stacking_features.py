"""Tree-based probability stacking for hybrid indoor positioning (Member 2).

This module implements Hybrid Approach A:
Extract class posterior probability distributions from classical tree ensembles (Random Forest)
for Building and Floor, and concatenate them with normalized RSSI features to form an augmented
representation for downstream neural network prediction.

Leakage Prevention Principle
----------------------------
To prevent target leakage during training:
1. Training set probabilities MUST be generated via Out-Of-Fold (OOF) cross-validation.
   A sample's probability vector is predicted strictly by tree models that NEVER saw that sample
   or its label during training.
2. Full tree classifiers are then fitted on the entire training set ONCE.
3. The full tree classifiers generate test and validation probabilities at inference time.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STACKING_PATH = REPO_ROOT / "models" / "tree_stacking_extractor.joblib"


class TreeProbabilityExtractor:
    """Extracts building and floor class probability features using tree ensembles."""

    def __init__(
        self,
        n_estimators: int = 150,
        min_samples_leaf: int = 1,
        n_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.rf_building: Optional[RandomForestClassifier] = None
        self.rf_floor: Optional[RandomForestClassifier] = None
        self.n_building_classes: Optional[int] = None
        self.n_floor_classes: Optional[int] = None

    def fit_transform_train(
        self, X_train: np.ndarray, building_train: np.ndarray, floor_train: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate out-of-fold (OOF) probability vectors for the training set without leakage,
        and fit the full classifiers on the entire training set.

        Returns:
            oof_building_probs: shape (N, n_buildings)
            oof_floor_probs:    shape (N, n_floors)
        """
        n_samples = len(X_train)
        self.n_building_classes = len(np.unique(building_train))
        self.n_floor_classes = len(np.unique(floor_train))

        oof_bld = np.zeros((n_samples, self.n_building_classes), dtype=np.float32)
        oof_flr = np.zeros((n_samples, self.n_floor_classes), dtype=np.float32)

        # Stratify by floor (finer spatial granularity ensures balanced building representation too)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, floor_train)):
            fold_bld_clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + fold,
                n_jobs=self.n_jobs,
            )
            fold_flr_clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + fold,
                n_jobs=self.n_jobs,
            )

            fold_bld_clf.fit(X_train[trn_idx], building_train[trn_idx])
            fold_flr_clf.fit(X_train[trn_idx], floor_train[trn_idx])

            oof_bld[val_idx] = fold_bld_clf.predict_proba(X_train[val_idx])
            oof_flr[val_idx] = fold_flr_clf.predict_proba(X_train[val_idx])

        # Fit full tree models on all training data for downstream validation/test inference
        self.rf_building = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.rf_floor = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.rf_building.fit(X_train, building_train)
        self.rf_floor.fit(X_train, floor_train)

        return oof_bld, oof_flr

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class probability vectors for unseen validation or test fingerprints."""
        if self.rf_building is None or self.rf_floor is None:
            raise RuntimeError("TreeProbabilityExtractor must be fitted before calling transform().")
        bld_probs = self.rf_building.predict_proba(X).astype(np.float32)
        flr_probs = self.rf_floor.predict_proba(X).astype(np.float32)
        return bld_probs, flr_probs

    def save(self, filepath: Path = DEFAULT_STACKING_PATH):
        """Save fitted probability extractor to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path = DEFAULT_STACKING_PATH) -> "TreeProbabilityExtractor":
        """Load fitted extractor from disk."""
        return joblib.load(filepath)


def concatenate_hybrid_features(
    rssi_features: np.ndarray,
    building_probabilities: np.ndarray,
    floor_probabilities: np.ndarray,
) -> np.ndarray:
    """Concatenate raw/normalized RSSI features with tree-derived probabilities.

    Output shape: (N, n_rssi + n_buildings + n_floors)
    """
    return np.hstack(
        [
            rssi_features,
            building_probabilities,
            floor_probabilities,
        ]
    ).astype(np.float32)
