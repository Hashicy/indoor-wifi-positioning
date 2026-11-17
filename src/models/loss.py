"""Loss functions for the multitask indoor positioning model (Member 3).

Implements:
1. BasicCombinedLoss  — L_total = L_building + L_floor + L_coords (baseline)
2. UncertaintyLoss    — Homoscedastic uncertainty-based task weighting
                        (Kendall, Gal & Cipolla, CVPR 2018)

Mathematical Justification
--------------------------
Building and floor use cross-entropy (classification losses) while coordinates
use MSE / SmoothL1 (regression loss). Their numerical scales can differ by
orders of magnitude. If one loss dominates the combined objective, the
optimizer effectively ignores the others.

Homoscedastic uncertainty weighting learns a task-specific log-variance
parameter s_i for each task. The weighted loss becomes:

    L_total = sum_i [ exp(-s_i) * L_i + c_i * s_i ]

For regression (Gaussian likelihood):  c_i = 0.5 (from log N(y; f(x), σ²))
For classification (softmax likelihood): c_i = 1.0 (from -log p(y|x))

exp(-s_i) acts as an adaptive precision weight: when a task's loss is
large, s_i grows, reducing that task's influence. The regularising term
c_i * s_i prevents s_i from diverging (which would zero out the task).
"""

import torch
import torch.nn as nn


class BasicCombinedLoss(nn.Module):
    """L_total = L_building + L_floor + L_coords (equal weights)."""

    def __init__(self, coord_loss="mse"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.coord_loss = coord_loss

    def forward(self, outputs, building, floor, coords):
        losses = {}
        if "building" in outputs:
            losses["building"] = self.ce(outputs["building"], building)
        if "floor" in outputs:
            losses["floor"] = self.ce(outputs["floor"], floor)
        if "coords" in outputs:
            if self.coord_loss == "smoothl1":
                losses["coords"] = nn.functional.smooth_l1_loss(outputs["coords"], coords)
            else:
                losses["coords"] = nn.functional.mse_loss(outputs["coords"], coords)
        losses["total"] = sum(losses.values())
        return losses


class UncertaintyLoss(nn.Module):
    """Homoscedastic uncertainty weighting (Kendall et al., 2018).

    Each task i gets a learnable log-variance parameter s_i = log(sigma_i^2).
    The weighted loss is:
        L_total = sum_i [ exp(-s_i) * L_i + c_i * s_i ]

    where c_i = 0.5 for regression and c_i = 1.0 for classification.

    The parameters s_i are optimised jointly with the network weights.
    """

    def __init__(self, coord_loss="mse"):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.coord_loss = coord_loss
        # Learnable log-variance for each task (initialised to 0 → weight=1)
        self.log_vars = nn.Parameter(torch.zeros(3))  # [building, floor, coords]

    def forward(self, outputs, building, floor, coords):
        raw_losses = {}
        if "building" in outputs:
            raw_losses["building"] = self.ce(outputs["building"], building)
        if "floor" in outputs:
            raw_losses["floor"] = self.ce(outputs["floor"], floor)
        if "coords" in outputs:
            if self.coord_loss == "smoothl1":
                raw_losses["coords"] = nn.functional.smooth_l1_loss(outputs["coords"], coords)
            else:
                raw_losses["coords"] = nn.functional.mse_loss(outputs["coords"], coords)

        # Uncertainty-weighted combination
        total = torch.tensor(0.0, device=coords.device)
        task_order = ["building", "floor", "coords"]
        # c_i: coefficient depends on the likelihood
        # classification (categorical/softmax) → c=1.0, regression (Gaussian) → c=0.5
        coefficients = {"building": 1.0, "floor": 1.0, "coords": 0.5}

        for i, task in enumerate(task_order):
            if task in raw_losses:
                precision = torch.exp(-self.log_vars[i])
                total = total + precision * raw_losses[task] + coefficients[task] * self.log_vars[i]

        losses = dict(raw_losses)
        losses["total"] = total
        # Expose learned weights for logging
        losses["_log_vars"] = self.log_vars.detach().clone()
        return losses

    def get_effective_weights(self):
        """Return the current exp(-s_i) precision weights for monitoring."""
        with torch.no_grad():
            return {
                "building_weight": torch.exp(-self.log_vars[0]).item(),
                "floor_weight": torch.exp(-self.log_vars[1]).item(),
                "coords_weight": torch.exp(-self.log_vars[2]).item(),
            }
