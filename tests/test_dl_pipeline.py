"""Tests for the deep learning data pipeline and network (Member 1).

Run with:  pytest tests/test_dl_pipeline.py
Uses a small made-up data set, so the real CSVs are not needed.
"""

import numpy as np
import pandas as pd
import torch

from src.data_loader import WAP_COLS
from src.data_preprocessing import (IMPUTE_VALUE, NOT_DETECTED, clean_training_data, fit_preprocessing,
                                    position_ids, split_train_val, transform)
from src.models.multitask_net import MultitaskNet


def fake_scans(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rssi = np.where(rng.random((n, len(WAP_COLS))) < 0.95, NOT_DETECTED, rng.integers(-104, -30, (n, len(WAP_COLS))))
    df = pd.DataFrame(rssi, columns=WAP_COLS)
    df.iloc[:, :10] = NOT_DETECTED                       # 10 APs never detected
    point = rng.integers(0, 40, n)                       # 40 positions, ~10 scans each
    df["BUILDINGID"] = point % 3
    df["FLOOR"] = point % 4
    df["LONGITUDE"] = -7600.0 + point * 5
    df["LATITUDE"] = 4864800.0 + point * 3
    return df


def test_cleaning_removes_duplicates_and_empty_scans():
    df = fake_scans()
    df = pd.concat([df, df.iloc[:5]])                    # 5 duplicates
    df.loc[df.index[0], WAP_COLS] = NOT_DETECTED         # make one scan empty
    clean = clean_training_data(df)
    assert not clean.duplicated().any()
    assert (clean[WAP_COLS] != NOT_DETECTED).any(axis=1).all()


def test_split_keeps_positions_on_one_side():
    train, val = split_train_val(fake_scans())
    assert set(position_ids(train)).isdisjoint(set(position_ids(val)))


def test_preprocessing_is_fitted_on_train_only():
    train, val = split_train_val(fake_scans())
    prep = fit_preprocessing(train)
    # never-detected APs are dropped
    assert all((train[w] != NOT_DETECTED).any() for w in prep["used_waps"])
    d_train, d_val = transform(train, prep), transform(val, prep)
    # same columns for both sets, inputs scaled to 0..1
    assert d_train["X"].shape[1] == d_val["X"].shape[1] == len(prep["used_waps"])
    assert d_train["X"].min() >= 0 and d_train["X"].max() <= 1
    assert d_val["X"].min() >= 0 and d_val["X"].max() <= 1
    # coordinates standardised with train statistics, and can be turned back into metres
    assert np.allclose(d_train["coords"].mean(axis=0), 0, atol=1e-5)
    back = prep["coord_scaler"].inverse_transform(d_val["coords"])
    assert np.allclose(back, d_val["coords_m"], atol=1e-2)


def test_not_detected_maps_to_minimum():
    train, _ = split_train_val(fake_scans())
    prep = fit_preprocessing(train)
    X = transform(train, prep)["X"]
    raw = train[prep["used_waps"]].to_numpy()
    # every "not detected" cell becomes the smallest value (0 after scaling)
    assert np.all(X[raw == NOT_DETECTED] == 0)
    assert IMPUTE_VALUE < -104


def test_network_output_shapes():
    net = MultitaskNet(n_inputs=50, n_buildings=3, n_floors=5)
    out = net(torch.randn(8, 50))
    assert out["building"].shape == (8, 3)
    assert out["floor"].shape == (8, 5)
    assert out["coords"].shape == (8, 2)


def test_single_task_network_has_one_head():
    net = MultitaskNet(n_inputs=50, tasks=("floor",))
    assert list(net(torch.randn(2, 50)).keys()) == ["floor"]
