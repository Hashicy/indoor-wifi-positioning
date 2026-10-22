"""Preprocessing and feature engineering pipeline for indoor Wi‑Fi positioning.

This module provides:
1. ``WifiPreprocessor`` – a scikit‑learn compatible transformer that:
   - Replaces the missing‑signal sentinel (100) with a plausible low RSSI
     value (default ``-110`` dBm).
   - Creates binary missingness indicator columns for each AP.
   - Generates aggregate features (``num_aps``, ``max_rssi``, ``mean_rssi``,
     ``std_rssi``, ``range_rssi``).
   - Optionally applies StandardScaler (fitted on training data only).
2. ``prepare_features()`` – end‑to‑end convenience function that loads data,
   separates X / y / meta, and returns transformed DataFrames.
3. ``build_preprocessing_pipeline()`` – factory function for ``WifiPreprocessor``.
4. ``get_feature_dictionary()`` – returns a markdown‑ready table of all features.
5. Ablation helper constants (``ABLATION_CONFIGS``) for systematic comparison.

Design decisions
----------------
- **Why −110?**  The lowest observed RSSI in the dataset is −104 dBm.
  Choosing −110 keeps the imputed value clearly below any real signal while
  remaining on the same numeric scale.
- **Why missingness indicators?**  Whether an AP is detected at all is a strong
  signal for coarse location (building / floor). A tree model can split on
  the binary indicator directly; a linear model benefits from the explicit
  encoding.
- **Why fit scaling on training only?**  To prevent data leakage from the
  validation (test) set into the training pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MISSING_SENTINEL = 100
DEFAULT_IMPUTE_VALUE = -110

# Ablation experiment configurations
ABLATION_CONFIGS: Dict[str, dict] = {
    "raw":              {"impute": False, "indicators": False, "aggregates": False, "scale": False},
    "imputed":          {"impute": True,  "indicators": False, "aggregates": False, "scale": False},
    "imputed+ind":      {"impute": True,  "indicators": True,  "aggregates": False, "scale": False},
    "imputed+ind+agg":  {"impute": True,  "indicators": True,  "aggregates": True,  "scale": False},
    "full_scaled":      {"impute": True,  "indicators": True,  "aggregates": True,  "scale": True},
}


# ──────────────────────────────────────────────────────────────────────────────
# Feature dictionary
# ──────────────────────────────────────────────────────────────────────────────
def get_feature_dictionary() -> pd.DataFrame:
    """Return a DataFrame documenting every engineered feature.

    This is intended for inclusion in the project report.
    """
    rows = [
        ("WAP001–WAP520", "RSSI", "Received signal strength from each of the 520 access "
         "points.  Original sentinel 100 (not detected) is replaced by −110 dBm.", "dBm"),
        ("WAP001_miss–WAP520_miss", "Indicator", "Binary flag: 1 if the AP was *not* "
         "detected in this scan, 0 if it was.", "0 / 1"),
        ("num_aps", "Aggregate", "Number of access points detected in this scan "
         "(count of WAP columns with original value ≠ 100).", "count"),
        ("max_rssi", "Aggregate", "Strongest RSSI among detected access points in "
         "this scan.", "dBm"),
        ("mean_rssi", "Aggregate", "Mean RSSI of the detected access points "
         "(excludes missing / imputed values).", "dBm"),
        ("std_rssi", "Aggregate", "Standard deviation of RSSI among detected APs.  "
         "Captures signal diversity.", "dBm"),
        ("range_rssi", "Aggregate", "Difference between the strongest and weakest "
         "detected RSSI.  Larger range may indicate open areas.", "dBm"),
    ]
    return pd.DataFrame(rows, columns=["Feature(s)", "Category", "Description", "Unit"])


# ──────────────────────────────────────────────────────────────────────────────
# Transformer
# ──────────────────────────────────────────────────────────────────────────────
class WifiPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocess raw Wi‑Fi RSSI measurements.

    Parameters
    ----------
    rssi_columns : list[str] | None
        Explicit list of RSSI column names.  If ``None`` (the default), the
        transformer infers them by selecting columns that start with ``"WAP"``.
    missing_sentinel : int, default 100
        Value used in the raw dataset to indicate a missing measurement.
    impute_value : int, default -110
        Replacement RSSI for the sentinel after preprocessing.
    add_indicators : bool, default True
        Whether to create binary missingness indicator columns.
    indicator_mode : {"missing", "observed"}, default "missing"
        If "missing", indicator is 1 if unobserved (sentinel), 0 if observed.
        If "observed", indicator is 1 if observed, 0 if unobserved (sentinel).
    add_aggregates : bool, default True
        Whether to generate aggregate features (num_aps, max_rssi, etc.).
    scale : bool, default True
        Whether to apply ``StandardScaler`` to the numeric features.
    drop_zero_variance : bool, default True
        Whether to drop WAP columns that are never detected in the training set.
    """

    def __init__(
        self,
        rssi_columns: Optional[List[str]] = None,
        missing_sentinel: int = DEFAULT_MISSING_SENTINEL,
        impute_value: int = DEFAULT_IMPUTE_VALUE,
        add_indicators: bool = True,
        indicator_mode: str = "missing",
        add_aggregates: bool = True,
        scale: bool = True,
        drop_zero_variance: bool = True,
    ):
        self.rssi_columns = rssi_columns
        self.missing_sentinel = missing_sentinel
        self.impute_value = impute_value
        self.add_indicators = add_indicators
        self.indicator_mode = indicator_mode
        self.add_aggregates = add_aggregates
        self.scale = scale
        self.drop_zero_variance = drop_zero_variance

    # ── helpers ──────────────────────────────────────────────────────────
    def _discover_rssi_columns(self, X: pd.DataFrame) -> List[str]:
        if self.rssi_columns is not None:
            return list(self.rssi_columns)
        return [c for c in X.columns if c.upper().startswith("WAP")]

    @staticmethod
    def _observed_mask(df: pd.DataFrame, cols: List[str], impute_value: int) -> pd.DataFrame:
        """Boolean mask: True where an AP was actually detected."""
        return df[cols] != impute_value

    # ── fit / transform ─────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y=None):
        """Learn column names, zero‑variance columns, and scaler parameters."""
        self._all_rssi_cols = self._discover_rssi_columns(X)

        # Identify columns never detected in training
        if self.drop_zero_variance:
            seen = (X[self._all_rssi_cols] != self.missing_sentinel).any()
            self._zero_var_cols = seen[~seen].index.tolist()
            self._rssi_cols = [c for c in self._all_rssi_cols if c not in self._zero_var_cols]
        else:
            self._zero_var_cols = []
            self._rssi_cols = list(self._all_rssi_cols)

        # Build a temporary transformed frame to discover numeric column names
        X_tmp = self._transform_core(X)
        self._output_columns = list(X_tmp.columns)
        self._numeric_features = self._get_numeric_col_names()

        if self.scale:
            self._scaler = StandardScaler()
            self._scaler.fit(X_tmp[self._numeric_features])
        else:
            self._scaler = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the full preprocessing pipeline."""
        if not hasattr(self, "_rssi_cols"):
            raise RuntimeError("Call fit() before transform().")
        out = self._transform_core(X)
        if self.scale and self._scaler is not None:
            out[self._numeric_features] = self._scaler.transform(out[self._numeric_features])
        return out

    def _transform_core(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform without scaling (used during fit to determine columns)."""
        out = pd.DataFrame(index=X.index)

        # 1. Replace sentinel
        rssi = X[self._rssi_cols].copy()
        original_sentinel_mask = rssi == self.missing_sentinel
        rssi = rssi.replace(self.missing_sentinel, self.impute_value)
        out[self._rssi_cols] = rssi

        # 2. Missingness / detection indicators
        if self.add_indicators:
            for c in self._rssi_cols:
                if self.indicator_mode == "observed":
                    out[f"{c}_miss"] = (~original_sentinel_mask[c]).astype(np.int8)
                else:
                    out[f"{c}_miss"] = original_sentinel_mask[c].astype(np.int8)

        # 3. Aggregate features
        if self.add_aggregates:
            observed = self._observed_mask(out, self._rssi_cols, self.impute_value)
            obs_vals = out[self._rssi_cols].where(observed)
            out["num_aps"]   = observed.sum(axis=1).astype(int)
            out["max_rssi"]  = obs_vals.max(axis=1)
            out["mean_rssi"] = obs_vals.mean(axis=1)
            out["std_rssi"]  = obs_vals.std(axis=1)
            out["range_rssi"] = obs_vals.max(axis=1) - obs_vals.min(axis=1)
            # Fill NaN for rows where nothing was observed
            for col in ["max_rssi", "mean_rssi", "std_rssi", "range_rssi"]:
                out[col] = out[col].fillna(self.impute_value if col != "range_rssi" else 0.0)
        return out

    def _get_numeric_col_names(self) -> List[str]:
        """Return the list of numeric columns that will be scaled."""
        cols = list(self._rssi_cols)
        if self.add_indicators:
            cols += [f"{c}_miss" for c in self._rssi_cols]
        if self.add_aggregates:
            cols += ["num_aps", "max_rssi", "mean_rssi", "std_rssi", "range_rssi"]
        return cols

    # ── public properties ────────────────────────────────────────────────
    @property
    def rssi_columns_(self) -> List[str]:
        """Columns actually used (after dropping zero‑variance if enabled)."""
        return getattr(self, "_rssi_cols", [])

    @property
    def dropped_columns_(self) -> List[str]:
        """Zero‑variance columns removed during fit."""
        return getattr(self, "_zero_var_cols", [])

    @property
    def numeric_features_(self) -> List[str]:
        """All numeric feature names in the output."""
        return getattr(self, "_numeric_features", [])

    @property
    def output_columns_(self) -> List[str]:
        """Full list of output column names."""
        return getattr(self, "_output_columns", [])

    @property
    def n_features_out_(self) -> int:
        return len(self.output_columns_)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────
def build_preprocessing_pipeline(
    rssi_columns: Optional[List[str]] = None,
    missing_sentinel: int = DEFAULT_MISSING_SENTINEL,
    impute_value: int = DEFAULT_IMPUTE_VALUE,
    add_indicators: bool = True,
    indicator_mode: str = "missing",
    add_aggregates: bool = True,
    scale: bool = True,
    drop_zero_variance: bool = True,
) -> WifiPreprocessor:
    """Create a fully configured ``WifiPreprocessor`` instance.

    Example::

        pipeline = build_preprocessing_pipeline()
        pipeline.fit(train_df[WAP_COLS])
        X_train = pipeline.transform(train_df[WAP_COLS])
    """
    return WifiPreprocessor(
        rssi_columns=rssi_columns,
        missing_sentinel=missing_sentinel,
        impute_value=impute_value,
        add_indicators=add_indicators,
        indicator_mode=indicator_mode,
        add_aggregates=add_aggregates,
        scale=scale,
        drop_zero_variance=drop_zero_variance,
    )


# ──────────────────────────────────────────────────────────────────────────────
# End‑to‑end helper
# ──────────────────────────────────────────────────────────────────────────────
def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    wap_cols: List[str],
    target_cols: List[str],
    meta_cols: List[str],
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, WifiPreprocessor]:
    """Separate features / targets / metadata and apply preprocessing.

    Parameters
    ----------
    train_df, val_df : raw DataFrames from data_loader.
    wap_cols, target_cols, meta_cols : column name lists.
    config : dict with keys ``impute``, ``indicators``, ``aggregates``,
             ``scale`` (booleans).  Defaults to the full pipeline.

    Returns
    -------
    X_train, X_val, y_train, y_val, meta_train, meta_val, preprocessor
    """
    if config is None:
        config = ABLATION_CONFIGS["full_scaled"]

    # Separate
    X_train_raw = train_df[wap_cols].copy()
    X_val_raw   = val_df[wap_cols].copy()
    y_train     = train_df[target_cols].copy()
    y_val       = val_df[target_cols].copy()
    meta_train  = train_df[[c for c in meta_cols if c in train_df.columns]].copy()
    meta_val    = val_df[[c for c in meta_cols if c in val_df.columns]].copy()

    # Build preprocessor from config
    preproc = WifiPreprocessor(
        rssi_columns=wap_cols,
        add_indicators=config.get("indicators", True),
        add_aggregates=config.get("aggregates", True),
        scale=config.get("scale", True),
        drop_zero_variance=True,
    )

    if not config.get("impute", True):
        # "raw" mode: just pass sentinel as‑is (still need the transformer for
        # consistency, but set impute_value = sentinel so nothing changes)
        preproc.impute_value = preproc.missing_sentinel

    preproc.fit(X_train_raw)
    X_train = preproc.transform(X_train_raw)
    X_val   = preproc.transform(X_val_raw)

    return X_train, X_val, y_train, y_val, meta_train, meta_val, preproc
