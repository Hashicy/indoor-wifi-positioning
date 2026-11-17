"""Data preparation for the deep learning phase (Member 1).

Every DL experiment in the project should get its data from prepare_data(), so that the whole team
uses the same split, the same scaling and the same label encoding.

The three data sets
-------------------
train : 85% of the cleaned training file          -> used to fit the model
val   : 15% of the cleaned training file          -> used for early stopping and choosing settings
test  : the official validationData.csv (1,111)   -> used ONCE, for final results

The train/val split is grouped by position: all scans taken at the same spot go to the same side.
Each position was scanned about 20 times (see notebook 01, section 6.5), so a random split would put
near-identical scans on both sides and make validation scores look far too good.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.data_loader import load_train, load_validation, WAP_COLS

REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_FILE = REPO_ROOT / "models" / "dl_preprocessing.joblib"

NOT_DETECTED = 100       # dataset code for "access point not detected"
IMPUTE_VALUE = -110      # below the weakest real reading (-104 dBm)
VAL_FRACTION = 0.15
RANDOM_SEED = 42


def clean_training_data(df):
    """Drop exact duplicates and scans that detected no access point (same as Phase 1)."""
    df = df.drop_duplicates()
    detected_any = (df[WAP_COLS] != NOT_DETECTED).any(axis=1)
    return df[detected_any].reset_index(drop=True)


def position_ids(df):
    """One label per unique position (building, floor, longitude, latitude)."""
    return (df["BUILDINGID"].astype(str) + "_" + df["FLOOR"].astype(str) + "_" +
            df["LONGITUDE"].astype(str) + "_" + df["LATITUDE"].astype(str))


def split_train_val(df):
    """Split the cleaned training file into train and val, keeping each position on one side."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(df, groups=position_ids(df)))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def fit_preprocessing(train_df):
    """Learn everything that depends on data, from the TRAIN part only."""
    # keep only access points that were detected at least once in train
    used_waps = [w for w in WAP_COLS if (train_df[w] != NOT_DETECTED).any()]

    rssi = train_df[used_waps].replace(NOT_DETECTED, IMPUTE_VALUE)
    rssi_scaler = MinMaxScaler().fit(rssi)                          # signals -> 0..1

    coords = train_df[["LONGITUDE", "LATITUDE"]]
    coord_scaler = StandardScaler().fit(coords)                     # metres -> mean 0, std 1

    building_encoder = LabelEncoder().fit(train_df["BUILDINGID"])
    floor_encoder = LabelEncoder().fit(train_df["FLOOR"])

    return {
        "used_waps": used_waps,
        "rssi_scaler": rssi_scaler,
        "coord_scaler": coord_scaler,
        "building_encoder": building_encoder,
        "floor_encoder": floor_encoder,
    }


def transform(df, prep):
    """Apply the fitted preprocessing to any data set (train, val or test)."""
    rssi = df[prep["used_waps"]].replace(NOT_DETECTED, IMPUTE_VALUE)
    X = prep["rssi_scaler"].transform(rssi)
    X = np.clip(X, 0, 1)            # a test reading stronger than any training one stays at 1

    return {
        "X": X.astype(np.float32),
        "building": prep["building_encoder"].transform(df["BUILDINGID"]),
        "floor": prep["floor_encoder"].transform(df["FLOOR"]),
        "coords": prep["coord_scaler"].transform(df[["LONGITUDE", "LATITUDE"]]).astype(np.float32),
        "coords_m": df[["LONGITUDE", "LATITUDE"]].to_numpy(),        # true coordinates in metres
    }


def prepare_data(save=True):
    """Load, clean, split, fit and transform. Returns (data, prep).

    data["train"], data["val"], data["test"] each hold X, building, floor, coords, coords_m.
    """
    full_train = clean_training_data(load_train())
    train_df, val_df = split_train_val(full_train)
    test_df = load_validation()

    prep = fit_preprocessing(train_df)
    data = {
        "train": transform(train_df, prep),
        "val": transform(val_df, prep),
        "test": transform(test_df, prep),
    }
    prep["sizes"] = {name: len(d["X"]) for name, d in data.items()}

    if save:
        PREPROCESSING_FILE.parent.mkdir(exist_ok=True)
        joblib.dump(prep, PREPROCESSING_FILE)
    return data, prep


def load_preprocessing():
    """Load the saved preprocessing objects (for inference on new scans)."""
    return joblib.load(PREPROCESSING_FILE)


if __name__ == "__main__":
    data, prep = prepare_data()
    print("Input features:", len(prep["used_waps"]))
    print("Sizes:", prep["sizes"])
    print("Buildings:", prep["building_encoder"].classes_.tolist(),
          " Floors:", prep["floor_encoder"].classes_.tolist())
