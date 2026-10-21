"""Load the UJIIndoorLoc training and validation data.
 
Usage
-----
    from src.data_loader import load_train, load_validation, split_columns, WAP_COLS
 
    train = load_train()
    X, y, meta = split_columns(train)
 
Run ``python -m src.data_loader`` from the repository root to check that the
data files are in place.
"""
 
from pathlib import Path
 
import pandas as pd
 
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TRAIN_FILE = DATA_DIR / "trainingData.csv"
VALIDATION_FILE = DATA_DIR / "validationData.csv"
 
# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
N_WAPS = 520
WAP_COLS = [f"WAP{i:03d}" for i in range(1, N_WAPS + 1)]
 
CLASSIFICATION_TARGETS = ["BUILDINGID", "FLOOR"]
REGRESSION_TARGETS = ["LONGITUDE", "LATITUDE"]
TARGET_COLS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS
 
# Kept for robustness / leakage analysis, never used as model inputs by default.
META_COLS = ["SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP"]
 
ALL_COLS = WAP_COLS + REGRESSION_TARGETS + ["FLOOR", "BUILDINGID"] + META_COLS
 
# Value used in the raw data when an access point was NOT detected.
MISSING_RSSI = 100
 
# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_DOWNLOAD_HINT = (
    "Download the dataset from "
    "https://archive.ics.uci.edu/dataset/310/ujiindoorloc and place "
    "trainingData.csv and validationData.csv directly inside the data/ folder. "
    "See data/README.md."
)
 
 
def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {_DOWNLOAD_HINT}")
    df = pd.read_csv(path)
    missing = set(ALL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {sorted(missing)[:10]}")
    return df
 
 
def load_train() -> pd.DataFrame:
    """Return the raw training data (19,937 rows x 529 columns)."""
    return _load(TRAIN_FILE)
 
 
def load_validation() -> pd.DataFrame:
    """Return the raw validation data (1,111 rows x 529 columns).
 
    This file is our held-out test set. Do not use it for tuning.
    """
    return _load(VALIDATION_FILE)
 
 
def split_columns(df: pd.DataFrame):
    """Split a raw dataframe into (X_wap, y, meta).
 
    X_wap : the 520 raw RSSI columns (sentinel 100 still present)
    y     : BUILDINGID, FLOOR, LONGITUDE, LATITUDE
    meta  : SPACEID, RELATIVEPOSITION, USERID, PHONEID, TIMESTAMP
    """
    return df[WAP_COLS].copy(), df[TARGET_COLS].copy(), df[META_COLS].copy()
 
 
if __name__ == "__main__":
    for name, loader in [("train", load_train), ("validation", load_validation)]:
        data = loader()
        X, y, meta = split_columns(data)
        detected = (X != MISSING_RSSI).sum(axis=1)
        print(
            f"{name:<10} rows={len(data):>6}  cols={data.shape[1]}  "
            f"WAPs/sample: mean={detected.mean():.1f}, min={detected.min()}, max={detected.max()}"
        )
    print("Data loaded successfully.")
