"""Data augmentations for training robustness (Member 3).

Implements two training-time augmentation strategies:

1. RSSI Noise Augmentation
   - Adds Gaussian noise to observed (non-sentinel) RSSI values
   - Simulates real-world signal fluctuations from multipath, body shadowing
   - Must NOT modify sentinel/undetected values (they represent physical absence)

2. Access-Point (AP) Dropout
   - Randomly masks a fraction of observed AP readings to the undetected level
   - Simulates scenarios where fewer access points are visible
   - Helps model learn from partial scan patterns

Both augmentations operate on the ALREADY PREPROCESSED data (MinMaxScaled to [0,1]),
so sentinel values appear as 0.0 in the scaled space.
"""

import torch


# In the MinMaxScaled space, the sentinel -110 dBm maps to 0.0
# Any value > 0.0 represents a detected access point
SENTINEL_SCALED = 0.0
_EPS = 1e-6  # tolerance for floating point comparison


class RSSINoise:
    """Add Gaussian noise to observed (non-sentinel) RSSI values.

    Parameters
    ----------
    std : float
        Standard deviation of the noise in the scaled [0,1] space.
        A std of 0.02 corresponds to roughly ±1–2 dBm in the original
        RSSI range, which is realistic for temporal signal fluctuation.
    """

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply noise only to detected APs (values > sentinel)."""
        mask = x > (SENTINEL_SCALED + _EPS)  # True for detected APs
        noise = torch.randn_like(x) * self.std
        # Apply noise only where AP was detected; clamp to valid range
        augmented = x + noise * mask.float()
        return torch.clamp(augmented, 0.0, 1.0)

    def __repr__(self):
        return f"RSSINoise(std={self.std})"


class APDropout:
    """Randomly mask a fraction of visible AP readings during training.

    For each sample, with probability `drop_rate`, each detected AP is
    independently set to the sentinel (undetected) value. This simulates
    sparser scans due to environmental changes or device differences.

    Parameters
    ----------
    drop_rate : float
        Probability of dropping each detected AP (0.0 to 1.0).
        The guide suggests testing 0.05, 0.10, and 0.15.
    """

    def __init__(self, drop_rate: float = 0.10):
        assert 0.0 <= drop_rate <= 1.0, "drop_rate must be in [0, 1]"
        self.drop_rate = drop_rate

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Drop random detected APs to sentinel value."""
        detected_mask = x > (SENTINEL_SCALED + _EPS)
        # Generate uniform random values; drop where < drop_rate
        drop_mask = torch.rand_like(x) < self.drop_rate
        # Only drop APs that were actually detected
        should_drop = detected_mask & drop_mask
        augmented = x.clone()
        augmented[should_drop] = SENTINEL_SCALED
        return augmented

    def __repr__(self):
        return f"APDropout(drop_rate={self.drop_rate})"


class ComposeAugmentation:
    """Apply a sequence of augmentations."""

    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            x = aug(x)
        return x

    def __repr__(self):
        return f"Compose([{', '.join(repr(a) for a in self.augmentations)}])"
