# Data

This project uses the **UJIIndoorLoc** dataset from the UCI Machine Learning Repository.
Raw data files are **not committed** to this repository (see `.gitignore`). Each team member
downloads them locally by following the steps below.

## Source

- **Dataset page:** https://archive.ics.uci.edu/dataset/310/ujiindoorloc
- **Direct download:** https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip (about 1.4 MB zipped)
- **DOI:** 10.24432/C5MS59
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0). Sharing and adaptation are allowed with appropriate credit.

## How to download

1. Download the zip from the link above.
2. Unzip it. The archive contains a folder named `UJIndoorLoc/` (note the spelling) with two CSV files.
3. Copy **both CSV files directly into this `data/` folder**, so the layout is:

```
data/
├── README.md
├── trainingData.csv
└── validationData.csv
```

4. Check that everything loads:

```bash
python -m src.data_loader
```

## Files

| File | Rows | Columns | Role in this project |
|---|---|---|---|
| `trainingData.csv` | 19,937 | 529 | Training / reference fingerprints |
| `validationData.csv` | 1,111 | 529 | Held-out evaluation set (the official test labels were never publicly released, so this file serves as our test set) |

The validation data was collected about four months after the training data, by different users and partly different phones.

## Columns

| Column(s) | Description | Values |
|---|---|---|
| `WAP001` … `WAP520` | RSSI of each of the 520 Wi-Fi access points | -104 (very weak) to 0 dBm; **100 means the AP was not detected** |
| `LONGITUDE` | Projected x-coordinate (metres, UTM-style) | about -7696 to -7300 |
| `LATITUDE` | Projected y-coordinate (metres, UTM-style) | about 4864745 to 4865017 |
| `FLOOR` | Floor within the building | 0 to 4 |
| `BUILDINGID` | Building | 0, 1, 2 |
| `SPACEID` | Room/space identifier | categorical (0 in validation) |
| `RELATIVEPOSITION` | 1 = inside the space, 2 = outside in front of the door | categorical (0 in validation) |
| `USERID` | Anonymised user who collected the sample | 1 to 18 in training; **0 for all validation rows** |
| `PHONEID` | Android device identifier | 0 to 24 |
| `TIMESTAMP` | UNIX time of capture | integer |

## Important conventions

- **Missing-signal sentinel:** `100` is not a strong signal. It is a placeholder for "not detected". Real RSSI values are always ≤ 0. Any statistic computed on raw WAP columns must handle this value explicitly.
- **Targets:** `BUILDINGID` and `FLOOR` (classification), and `LONGITUDE` and `LATITUDE` (regression).
- **Not model inputs by default:** `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID` and `TIMESTAMP`. These are kept for robustness and leakage analysis only.
- **Never tune on `validationData.csv`.** Model selection uses the training file only (for example, grouped splits by `USERID`).

## Citation

Torres-Sospedra, J., Montoliu, R., Martínez-Usó, A., Avariento, J. P., Arnau, T. J., Benedito-Bordonau, M., & Huerta, J. (2014).
UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems.
*Proceedings of the International Conference on Indoor Positioning and Indoor Navigation (IPIN)*, 261–270.

---

## Preprocessed data (`data/` and `data/processed/`)

The preprocessed CSV files are generated using the recommended `imputed+ind+agg` feature pipeline.

### Files produced

| File | Location | Shape | Description |
|---|---|---|---|
| `preprocessed_train.csv` | `data/` | 19,227 × 939 | Combined training features + targets for direct model training |
| `preprocessed_val.csv` | `data/` | 1,111 × 939 | Combined validation features + targets for evaluation |
| `preprocessed_dataset.zip` | `data/` | ~2.1 MB | Compressed archive of all preprocessed CSVs for easy 1-click download |
| `X_train.csv` | `data/processed/` | 19,227 × 935 | Separated training features (imputed RSSI + indicators + 5 aggregates) |
| `y_train.csv` | `data/processed/` | 19,227 × 4 | Training targets: `BUILDINGID`, `FLOOR`, `LONGITUDE`, `LATITUDE` |
| `meta_train.csv` | `data/processed/` | 19,227 × 5 | Training metadata: `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, `TIMESTAMP` |
| `X_val.csv` | `data/processed/` | 1,111 × 935 | Separated validation features (same pipeline fitted on training only) |
| `y_val.csv` | `data/processed/` | 1,111 × 4 | Validation targets |
| `meta_val.csv` | `data/processed/` | 1,111 × 5 | Validation metadata |
| `feature_dictionary.csv` | `data/processed/` | — | Description of every feature column category |

> **Note:** Training rows drop from 19,937 (raw) to 19,227 (clean) because `prepare_features()`
> removes 637 exact duplicate rows and 73 empty scans (0 APs detected) as part of its
> preprocessing step.

### How to generate

You can generate all preprocessed CSV files in two ways:

1. **Via Python script (fastest, standalone):**
   ```bash
   python scripts/generate_preprocessed_data.py
   ```
2. **Via Jupyter notebook:**
   Open and run `notebooks/02_preprocessing_features.ipynb` (Section 12), or execute:
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/02_preprocessing_features.ipynb
   ```

### Loading preprocessed data in Python

**Option A: Direct load (combined features + targets):**
```python
import pandas as pd

train = pd.read_csv("data/preprocessed_train.csv")
val = pd.read_csv("data/preprocessed_val.csv")

# Targets
target_cols = ["BUILDINGID", "FLOOR", "LONGITUDE", "LATITUDE"]
X_train = train.drop(columns=target_cols)
y_train = train[target_cols]

X_val = val.drop(columns=target_cols)
y_val = val[target_cols]
```

**Option B: Separated splits:**
```python
import pandas as pd

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
meta_train = pd.read_csv("data/processed/meta_train.csv")

X_val = pd.read_csv("data/processed/X_val.csv")
y_val = pd.read_csv("data/processed/y_val.csv")
meta_val = pd.read_csv("data/processed/meta_val.csv")
```

If you need scaling (e.g. for k-NN or neural networks), fit a `StandardScaler` **on
`X_train` only** and apply it to both splits:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
```
