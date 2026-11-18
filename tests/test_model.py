"""Unit tests for Deep Learning & Hybrid Models (Member 5 Integration).

Covers:
- MultitaskNet (shared trunk, multi-head outputs, single-task mode)
- HybridMultitaskModel (probability stacking)
- UncertaintyLoss (homoscedastic loss weighting)
- Model checkpoint serialization and deserialization

Run with:
    pytest tests/test_model.py
"""

import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from src.models.multitask_net import MultitaskNet
from src.models.hybrid_model import HybridPositioningSystem
from src.models.loss import UncertaintyLoss


def test_multitask_net_shapes():
    batch_size = 16
    n_inputs = 459
    n_buildings = 3
    n_floors = 5

    model = MultitaskNet(n_inputs=n_inputs, n_buildings=n_buildings, n_floors=n_floors)
    x = torch.randn(batch_size, n_inputs)
    out = model(x)

    assert "building" in out
    assert "floor" in out
    assert "coords" in out
    assert out["building"].shape == (batch_size, n_buildings)
    assert out["floor"].shape == (batch_size, n_floors)
    assert out["coords"].shape == (batch_size, 2)


def test_multitask_net_single_task_modes():
    n_inputs = 50
    for task, expected_dim in [("building", 3), ("floor", 5), ("coords", 2)]:
        model = MultitaskNet(n_inputs=n_inputs, tasks=(task,))
        x = torch.randn(4, n_inputs)
        out = model(x)
        assert list(out.keys()) == [task]
        assert out[task].shape == (4, expected_dim)


def test_hybrid_stacked_model():
    batch_size = 8
    n_rssi = 459
    n_buildings = 3
    n_floors = 5
    # Total input: 459 + 3 + 5 = 467
    n_stacked_inputs = n_rssi + n_buildings + n_floors

    # The stacked neural net is a MultitaskNet with (n_rssi + 3 + 5) inputs
    model = MultitaskNet(n_inputs=n_stacked_inputs, n_buildings=n_buildings, n_floors=n_floors)
    x_stacked = torch.randn(batch_size, n_stacked_inputs)
    out = model(x_stacked)

    assert out["building"].shape == (batch_size, 3)
    assert out["floor"].shape == (batch_size, 5)
    assert out["coords"].shape == (batch_size, 2)

    # HybridPositioningSystem wrapper initialization
    system = HybridPositioningSystem(mode="full_hybrid", dl_model=model)
    assert system.mode == "full_hybrid"
    assert system.dl_model is not None


def test_uncertainty_loss_gradients():
    criterion = UncertaintyLoss(coord_loss="mse")
    outputs = {
        "building": torch.randn(4, 3, requires_grad=True),
        "floor": torch.randn(4, 5, requires_grad=True),
        "coords": torch.randn(4, 2, requires_grad=True),
    }
    bldg_target = torch.tensor([0, 1, 2, 0])
    floor_target = torch.tensor([0, 1, 2, 3])
    coords_target = torch.randn(4, 2)

    loss_dict = criterion(outputs, bldg_target, floor_target, coords_target)
    total_loss = loss_dict["total"]
    assert total_loss.item() > 0
    total_loss.backward()

    assert criterion.log_vars.grad is not None
    assert criterion.log_vars.grad.shape == (3,)


def test_model_save_and_load():
    model = MultitaskNet(n_inputs=459, n_buildings=3, n_floors=5)
    x = torch.randn(2, 459)
    model.eval()
    with torch.no_grad():
        orig_out = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pth"
        torch.save({"model_state_dict": model.state_dict(), "n_inputs": 459}, ckpt_path)

        loaded_data = torch.load(ckpt_path, weights_only=False)
        loaded_model = MultitaskNet(n_inputs=loaded_data["n_inputs"], n_buildings=3, n_floors=5)
        loaded_model.load_state_dict(loaded_data["model_state_dict"])
        loaded_model.eval()

        with torch.no_grad():
            loaded_out = loaded_model(x)

        for task in ["building", "floor", "coords"]:
            assert torch.allclose(orig_out[task], loaded_out[task], atol=1e-6)
