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
