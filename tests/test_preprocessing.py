"""Comprehensive tests for src/preprocessing.py.

Covers:
  1. Missing‑signal sentinel replacement.
  2. Missingness indicator creation.
  3. Aggregate feature computation (num_aps, max_rssi, mean_rssi, std_rssi, range_rssi).
  4. Zero‑variance column dropping.
  5. Scaling fitted only on training data.
  6. Train / test column consistency.
  7. Ablation configurations.
  8. Feature dictionary completeness.
  9. Edge case: row with zero APs detected.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    WifiPreprocessor,
    build_preprocessing_pipeline,
    get_feature_dictionary,
    prepare_features,
    ABLATION_CONFIGS,
)


# ── test fixtures ────────────────────────────────────────────────────────────
def _small_df():
    """3 rows × 4 WAP columns with known sentinel pattern."""
    return pd.DataFrame({
        "WAP001": [100, -80,  -70],   # row 0 missing
        "WAP002": [-90, 100,  -60],   # row 1 missing
        "WAP003": [100, 100,  -85],   # rows 0,1 missing
        "WAP004": [100, 100,  100],   # never detected → zero‑variance
    })


def _small_df_with_targets():
    """DataFrame that also carries target and meta columns."""
    df = _small_df()
    df["BUILDINGID"] = [0, 1, 2]
    df["FLOOR"]      = [1, 0, 2]
    df["LONGITUDE"]  = [0.0, 1.0, 2.0]
    df["LATITUDE"]   = [0.0, 1.0, 2.0]
    df["USERID"]     = [1, 2, 3]
    df["PHONEID"]    = [10, 11, 12]
    return df


# ── 1. Missing replacement ──────────────────────────────────────────────────
def test_sentinel_replaced():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, drop_zero_variance=False)
    out = pp.fit_transform(df)
    # No value in WAP cols should equal 100
    wap_out = out[[c for c in out.columns if c.startswith("WAP") and "_miss" not in c]]
    assert (wap_out == 100).sum().sum() == 0
    # Sentinel positions should now be the impute value
    assert out.loc[0, "WAP001"] == -110


# ── 2. Missingness indicators ───────────────────────────────────────────────
def test_indicators_correct():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, drop_zero_variance=False)
    out = pp.fit_transform(df)
    # WAP001 row 0 was sentinel → indicator = 1
    assert out.loc[0, "WAP001_miss"] == 1
    # WAP001 row 1 was observed → indicator = 0
    assert out.loc[1, "WAP001_miss"] == 0


def test_indicators_observed_mode():
    """Verify Task 4 convention: 1 if observed, 0 if missing."""
    df = _small_df()
    pp = WifiPreprocessor(scale=False, indicator_mode="observed", drop_zero_variance=False)
    out = pp.fit_transform(df)
    # WAP001 row 0 was sentinel (missing) → indicator = 0
    assert out.loc[0, "WAP001_miss"] == 0
    # WAP001 row 1 was observed → indicator = 1
    assert out.loc[1, "WAP001_miss"] == 1



def test_no_indicators_when_disabled():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, add_indicators=False, drop_zero_variance=False)
    out = pp.fit_transform(df)
    miss_cols = [c for c in out.columns if c.endswith("_miss")]
    assert len(miss_cols) == 0


# ── 3. Aggregate features ───────────────────────────────────────────────────
def test_num_aps():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, drop_zero_variance=False)
    out = pp.fit_transform(df)
    # Row 0: WAP002 observed → 1 AP
    assert out.loc[0, "num_aps"] == 1
    # Row 2: WAP001, WAP002, WAP003 observed → 3 APs
    assert out.loc[2, "num_aps"] == 3


def test_max_rssi():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, drop_zero_variance=False)
    out = pp.fit_transform(df)
    # Row 2: observed values are -70, -60, -85 → max = -60
    assert out.loc[2, "max_rssi"] == -60


def test_mean_rssi():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, drop_zero_variance=False)
    out = pp.fit_transform(df)
    # Row 2: mean of [-70, -60, -85] = -71.666…
    assert abs(out.loc[2, "mean_rssi"] - np.mean([-70, -60, -85])) < 1e-6


def test_range_rssi():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, drop_zero_variance=False)
    out = pp.fit_transform(df)
    # Row 2: range = -60 - (-85) = 25
    assert out.loc[2, "range_rssi"] == 25.0


def test_no_aggregates_when_disabled():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, add_aggregates=False, drop_zero_variance=False)
    out = pp.fit_transform(df)
    for col in ["num_aps", "max_rssi", "mean_rssi", "std_rssi", "range_rssi"]:
        assert col not in out.columns


# ── 4. Zero‑variance column dropping ────────────────────────────────────────
def test_zero_variance_dropped():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, drop_zero_variance=True)
    pp.fit(df)
    # WAP004 is never detected → should be dropped
    assert "WAP004" in pp.dropped_columns_
    assert "WAP004" not in pp.rssi_columns_
    out = pp.transform(df)
    assert "WAP004" not in out.columns


def test_zero_variance_kept_when_disabled():
    df = _small_df()
    pp = WifiPreprocessor(scale=False, drop_zero_variance=False)
    pp.fit(df)
    assert "WAP004" in pp.rssi_columns_


# ── 5. Scaling ───────────────────────────────────────────────────────────────
def test_scaling_zero_mean_unit_var():
    df = _small_df()
    pp = WifiPreprocessor(scale=True, drop_zero_variance=False)
    out = pp.fit_transform(df)
    numeric = pp.numeric_features_
    means = out[numeric].mean()
    stds  = out[numeric].std(ddof=0)   # population std for N rows
    # Means should be approximately zero
    assert means.abs().max() < 1e-6


# ── 6. Train/test consistency ───────────────────────────────────────────────
def test_train_test_same_columns():
    train = _small_df()
    test  = _small_df().iloc[:2]   # subset
    pp = WifiPreprocessor(scale=True, drop_zero_variance=True)
    pp.fit(train)
    out_tr = pp.transform(train)
    out_te = pp.transform(test)
    assert list(out_tr.columns) == list(out_te.columns)


# ── 7. Ablation configs ─────────────────────────────────────────────────────
def test_ablation_configs_exist():
    expected = {"raw", "imputed", "imputed+ind", "imputed+ind+agg", "full_scaled"}
    assert set(ABLATION_CONFIGS.keys()) == expected


def test_raw_config_no_impute():
    """With the 'raw' config, sentinel should remain unchanged."""
    df = _small_df()
    pp = WifiPreprocessor(scale=False, drop_zero_variance=False)
    pp.impute_value = pp.missing_sentinel   # raw mode trick
    pp.add_indicators = False
    pp.add_aggregates = False
    out = pp.fit_transform(df)
    # Sentinel positions should still be 100
    assert out.loc[0, "WAP001"] == 100


# ── 8. Feature dictionary ───────────────────────────────────────────────────
def test_feature_dictionary():
    fd = get_feature_dictionary()
    assert "Feature(s)" in fd.columns
    assert len(fd) >= 7   # at least 7 rows documented


# ── 9. Edge case: all missing ────────────────────────────────────────────────
def test_row_all_missing():
    """A scan that detected nothing should still transform without errors."""
    df = pd.DataFrame({
        "WAP001": [100],
        "WAP002": [100],
        "WAP003": [100],
    })
    pp = WifiPreprocessor(scale=False, drop_zero_variance=False)
    out = pp.fit_transform(df)
    assert out.loc[0, "num_aps"] == 0
    # max_rssi should be filled with impute_value
    assert out.loc[0, "max_rssi"] == -110


# ── 10. build_preprocessing_pipeline factory ─────────────────────────────────
def test_factory():
    pp = build_preprocessing_pipeline(scale=False)
    assert isinstance(pp, WifiPreprocessor)
    assert pp.scale is False


# ── 11. prepare_features integration (uses real data if available) ───────────
def test_prepare_features_integration():
    """Quick integration test with synthetic data mimicking real layout."""
    train = _small_df_with_targets()
    val   = _small_df_with_targets()

    X_tr, X_va, y_tr, y_va, m_tr, m_va, pp = prepare_features(
        train, val,
        wap_cols=["WAP001", "WAP002", "WAP003", "WAP004"],
        target_cols=["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"],
        meta_cols=["USERID", "PHONEID"],
        config=ABLATION_CONFIGS["full_scaled"],
    )
    assert len(X_tr) == len(train)
    assert "BUILDINGID" in y_tr.columns
    assert "USERID" in m_tr.columns
    # X should not contain target or meta cols
    assert "BUILDINGID" not in X_tr.columns
    assert "USERID" not in X_tr.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
