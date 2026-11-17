"""Tests for Member 3's loss functions, augmentations and ablation components."""

import pytest
import torch
import numpy as np

from src.models.loss import BasicCombinedLoss, UncertaintyLoss
from src.augmentations import RSSINoise, APDropout, ComposeAugmentation


# ──────────────────────────────────────────────────────────────────────────────
# Loss function tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBasicCombinedLoss:
    def setup_method(self):
        self.loss_fn = BasicCombinedLoss(coord_loss="mse")
        self.outputs = {
            "building": torch.randn(8, 3),
            "floor": torch.randn(8, 5),
            "coords": torch.randn(8, 2),
        }
        self.building = torch.randint(0, 3, (8,))
        self.floor = torch.randint(0, 5, (8,))
        self.coords = torch.randn(8, 2)

    def test_returns_all_loss_components(self):
        losses = self.loss_fn(self.outputs, self.building, self.floor, self.coords)
        assert "building" in losses
        assert "floor" in losses
        assert "coords" in losses
        assert "total" in losses

    def test_total_equals_sum(self):
        losses = self.loss_fn(self.outputs, self.building, self.floor, self.coords)
        expected = losses["building"] + losses["floor"] + losses["coords"]
        assert torch.allclose(losses["total"], expected, atol=1e-6)

    def test_all_losses_positive(self):
        losses = self.loss_fn(self.outputs, self.building, self.floor, self.coords)
        for name, val in losses.items():
            assert val.item() > 0, f"Loss '{name}' should be positive"

    def test_smoothl1_variant(self):
        loss_fn = BasicCombinedLoss(coord_loss="smoothl1")
        losses = loss_fn(self.outputs, self.building, self.floor, self.coords)
        assert "total" in losses


class TestUncertaintyLoss:
    def setup_method(self):
        self.loss_fn = UncertaintyLoss(coord_loss="mse")
        self.outputs = {
            "building": torch.randn(8, 3),
            "floor": torch.randn(8, 5),
            "coords": torch.randn(8, 2),
        }
        self.building = torch.randint(0, 3, (8,))
        self.floor = torch.randint(0, 5, (8,))
        self.coords = torch.randn(8, 2)

    def test_returns_all_components(self):
        losses = self.loss_fn(self.outputs, self.building, self.floor, self.coords)
        assert "building" in losses
        assert "floor" in losses
        assert "coords" in losses
        assert "total" in losses
        assert "_log_vars" in losses

    def test_has_learnable_parameters(self):
        params = list(self.loss_fn.parameters())
        assert len(params) == 1  # log_vars
        assert params[0].shape == (3,)

    def test_log_vars_initialized_to_zero(self):
        assert torch.allclose(self.loss_fn.log_vars.data, torch.zeros(3))

    def test_effective_weights_start_at_one(self):
        weights = self.loss_fn.get_effective_weights()
        for name, w in weights.items():
            assert abs(w - 1.0) < 1e-5, f"{name} should start at 1.0"

    def test_gradient_flows_to_log_vars(self):
        losses = self.loss_fn(self.outputs, self.building, self.floor, self.coords)
        losses["total"].backward()
        assert self.loss_fn.log_vars.grad is not None
        assert not torch.all(self.loss_fn.log_vars.grad == 0)

    def test_total_differs_from_basic_sum(self):
        """With non-zero log_vars, uncertainty loss should differ from simple sum."""
        self.loss_fn.log_vars.data = torch.tensor([0.5, -0.3, 0.1])
        losses = self.loss_fn(self.outputs, self.building, self.floor, self.coords)
        basic_sum = losses["building"] + losses["floor"] + losses["coords"]
        assert not torch.allclose(losses["total"], basic_sum, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRSSINoise:
    def test_does_not_modify_sentinels(self):
        """Sentinel values (0.0 = undetected AP) should remain untouched."""
        x = torch.zeros(100, 10)  # All sentinels
        aug = RSSINoise(std=0.05)
        result = aug(x)
        assert torch.allclose(result, x), "Sentinel values should not be modified"

    def test_modifies_detected_values(self):
        """Non-zero values (detected APs) should be noised."""
        torch.manual_seed(42)
        x = torch.ones(100, 10) * 0.5  # All detected
        aug = RSSINoise(std=0.05)
        result = aug(x)
        assert not torch.allclose(result, x), "Detected values should be modified"

    def test_output_in_valid_range(self):
        """Output should be clamped to [0, 1]."""
        torch.manual_seed(42)
        x = torch.rand(100, 10)
        aug = RSSINoise(std=0.1)
        result = aug(x)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_mixed_sentinel_and_detected(self):
        """Only detected values should change, sentinels preserved."""
        torch.manual_seed(42)
        x = torch.tensor([[0.0, 0.5, 0.0, 0.8, 0.0]])
        aug = RSSINoise(std=0.05)
        result = aug(x)
        # Sentinel columns should be 0
        assert result[0, 0].item() == 0.0
        assert result[0, 2].item() == 0.0
        assert result[0, 4].item() == 0.0


class TestAPDropout:
    def test_does_not_create_new_detections(self):
        """AP dropout should only mask, never create new detections."""
        x = torch.zeros(100, 10)  # All sentinels
        aug = APDropout(drop_rate=0.5)
        result = aug(x)
        assert torch.allclose(result, x)

    def test_drops_detected_aps(self):
        """Some detected APs should be masked to sentinel."""
        torch.manual_seed(42)
        x = torch.ones(1000, 50) * 0.5  # All detected
        aug = APDropout(drop_rate=0.5)
        result = aug(x)
        n_dropped = (result == 0.0).sum().item()
        # With 50% drop rate on 50,000 values, expect ~25,000 dropped
        assert n_dropped > 10000, "Should drop a substantial number of APs"
        assert n_dropped < 40000, "Should not drop all APs"

    def test_drop_rate_zero_no_change(self):
        x = torch.rand(100, 10)
        aug = APDropout(drop_rate=0.0)
        result = aug(x)
        assert torch.allclose(result, x)

    def test_drop_rate_one_drops_all_detected(self):
        x = torch.ones(100, 10) * 0.5
        aug = APDropout(drop_rate=1.0)
        result = aug(x)
        assert torch.all(result == 0.0)


class TestComposeAugmentation:
    def test_applies_both(self):
        torch.manual_seed(42)
        x = torch.ones(100, 10) * 0.5
        compose = ComposeAugmentation([RSSINoise(std=0.05), APDropout(drop_rate=0.1)])
        result = compose(x)
        # Should have some zeros (from AP dropout) and some noised values
        has_zeros = (result == 0.0).any().item()
        has_non_half = not torch.allclose(result[result > 0], torch.tensor(0.5))
        assert has_zeros or has_non_half, "Composition should modify the data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
