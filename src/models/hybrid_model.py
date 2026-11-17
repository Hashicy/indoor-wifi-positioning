"""Unified Hybrid Positioning Architecture combining Deep Learning with Tree Ensembles (Member 2).

This module defines the end-to-end `HybridPositioningSystem` which unifies:
1. Classical Random Forest building and floor class posterior probability extraction (Stacking).
2. Deep Multitask Neural Network with shared representation learning.
3. Gradient Boosted Decision Tree (GBDT) residual coordinate correction.

Supported Operational Modes
---------------------------
- 'dl_only'     : Baseline Multitask Deep Learning model.
- 'stacking'    : Multitask DL fed with concatenated RSSI + Tree class probabilities.
- 'residual'    : Multitask DL with GBDT residual spatial correction.
- 'full_hybrid' : Stacked Multitask DL + GBDT residual spatial correction.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import torch

from src.models.multitask_net import MultitaskNet
from src.residual_correction import ResidualCoordinateCorrector
from src.stacking_features import TreeProbabilityExtractor, concatenate_hybrid_features

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HybridPositioningSystem:
    """Unified hybrid model for indoor localization combining tree ensembles and deep neural networks."""

    def __init__(
        self,
        mode: str = "full_hybrid",
        dl_model: Optional[MultitaskNet] = None,
        tree_extractor: Optional[TreeProbabilityExtractor] = None,
        residual_corrector: Optional[ResidualCoordinateCorrector] = None,
    ):
        if mode not in ("dl_only", "stacking", "residual", "full_hybrid"):
            raise ValueError(f"Unknown hybrid mode '{mode}'.")
        self.mode = mode
        self.dl_model = dl_model
        self.tree_extractor = tree_extractor
        self.residual_corrector = residual_corrector

    @torch.no_grad()
    def predict(
        self,
        X_rssi: np.ndarray,
        prep: Dict[str, Any],
        device: torch.device = DEVICE,
    ) -> Dict[str, np.ndarray]:
        """Perform end-to-end inference on raw/preprocessed RSSI features.

        Parameters:
            X_rssi : Preprocessed RSSI array of shape (N, n_waps).
            prep   : Preprocessing dictionary containing encoders and scalers.
            device : PyTorch device.

        Returns:
            Dictionary containing:
                'building' : Predicted building IDs (original label space).
                'floor'    : Predicted floor numbers (original label space).
                'coords_m' : Predicted (x, y) continuous metric coordinates in metres.
        """
        if self.dl_model is None:
            raise RuntimeError("Deep learning model is not initialized.")

        self.dl_model.eval()
        self.dl_model.to(device)

        # 1. Feature Stacking Stage
        if self.mode in ("stacking", "full_hybrid"):
            if self.tree_extractor is None:
                raise RuntimeError("Tree probability extractor is required for stacking mode.")
            bld_probs, flr_probs = self.tree_extractor.transform(X_rssi)
            X_input = concatenate_hybrid_features(X_rssi, bld_probs, flr_probs)
        else:
            X_input = X_rssi

        # 2. Deep Multitask Forward Pass
        tensor_in = torch.tensor(X_input, dtype=torch.float32).to(device)
        outputs = self.dl_model(tensor_in)

        predictions: Dict[str, np.ndarray] = {}

        # Decode building prediction
        if "building" in outputs:
            bld_idx = outputs["building"].argmax(dim=1).cpu().numpy()
            predictions["building"] = prep["building_encoder"].inverse_transform(bld_idx)

        # Decode floor prediction
        if "floor" in outputs:
            flr_idx = outputs["floor"].argmax(dim=1).cpu().numpy()
            predictions["floor"] = prep["floor_encoder"].inverse_transform(flr_idx)

        # Decode baseline DL coordinates (scaled -> metres)
        if "coords" in outputs:
            coords_std = outputs["coords"].cpu().numpy()
            coords_dl_m = prep["coord_scaler"].inverse_transform(coords_std)
            predictions["coords_dl_m"] = coords_dl_m

            # 3. Residual GBDT Correction Stage
            if self.mode in ("residual", "full_hybrid") and self.residual_corrector is not None:
                corrected_m = self.residual_corrector.correct_coordinates(X_input, coords_dl_m)
                predictions["coords_m"] = corrected_m
            else:
                predictions["coords_m"] = coords_dl_m

        return predictions

    def save(self, filepath: Union[str, Path]):
        """Serialize complete hybrid pipeline metadata and non-PyTorch components."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "mode": self.mode,
            "tree_extractor": self.tree_extractor,
            "residual_corrector": self.residual_corrector,
            "dl_state_dict": self.dl_model.state_dict() if self.dl_model is not None else None,
            "dl_tasks": self.dl_model.tasks if self.dl_model is not None else None,
            "dl_n_inputs": self.dl_model.trunk[0][0].in_features if self.dl_model is not None else None,
            "dl_hidden": (
                self.dl_model.trunk[0][0].out_features,
                self.dl_model.trunk[1][0].out_features,
                self.dl_model.trunk[2][0].out_features,
            ) if self.dl_model is not None else None,
        }
        joblib.dump(bundle, filepath)

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        device: torch.device = DEVICE,
    ) -> "HybridPositioningSystem":
        """Deserialize and reconstruct the complete hybrid positioning system."""
        filepath = Path(filepath)
        bundle = joblib.load(filepath)

        dl_model = None
        if bundle["dl_state_dict"] is not None:
            dl_model = MultitaskNet(
                n_inputs=bundle["dl_n_inputs"],
                hidden=bundle.get("dl_hidden", (512, 256, 128)),
                tasks=bundle["dl_tasks"],
            )
            dl_model.load_state_dict(bundle["dl_state_dict"])
            dl_model.to(device)

        return cls(
            mode=bundle["mode"],
            dl_model=dl_model,
            tree_extractor=bundle["tree_extractor"],
            residual_corrector=bundle["residual_corrector"],
        )
