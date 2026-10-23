# Paper Notes: UJIIndoorLoc (Torres-Sospedra et al., 2014)

## Bibliographic Details

| Field | Value |
|-------|-------|
| **Title** | UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems |
| **Authors** | Joaquín Torres-Sospedra, Raul Montoliu, Adolfo Martínez-Usó, Joan P. Avariento, Tomas J. Arnau, Mauri Benedito-Bordonau, Joaquín Huerta |
| **Venue** | International Conference on Indoor Positioning and Indoor Navigation (IPIN 2014) |
| **Year** | 2014 |
| **DOI** | [10.1109/IPIN.2014.7275492](https://doi.org/10.1109/IPIN.2014.7275492) |
| **Dataset URL** | [UCI ML Repository – UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujindoorloc) |

---

## Motivation

- Indoor localization is an open problem because GPS signals are unreliable indoors.
- WLAN/WiFi fingerprinting is a promising approach because WiFi infrastructure is widely deployed.
- **Key gap addressed:** Prior to this paper, there was no common public benchmark database for comparing WiFi fingerprinting methods across multiple buildings and floors.

---

## Dataset Description (Verified from UCI Repository)

### Scale
- **Coverage:** 3 buildings at Universitat Jaume I, ~110,000 m² total
- **Floors:** Up to 4+ floors per building (Floor values: 0–4)
- **Training samples:** 19,937 records (`trainingData.csv`)
- **Validation samples:** 1,111 records (`validationData.csv`)
- **Total instances listed on UCI:** 21,048

### Attributes (529 total)
| Attribute Range | Description |
|----------------|-------------|
| WAP001–WAP520 (Attr 1–520) | RSSI intensity values. Negative integers from −104 dBm (very weak) to 0 dBm (strongest). Value **+100** indicates WAP not detected. |
| Longitude (Attr 521) | Negative real values (local coordinate system, not WGS84 degrees) |
| Latitude (Attr 522) | Positive real values (local coordinate system) |
| Floor (Attr 523) | Integer 0–4 |
| BuildingID (Attr 524) | Integer 0–2 (3 buildings) |
| SpaceID (Attr 525) | Internal space identifier (office, corridor, classroom) |
| RelativePosition (Attr 526) | 1 = inside space, 2 = outside in front of door |
| UserID (Attr 527) | Anonymized user ID (0–18; user 0 is the validation user) |
| PhoneID (Attr 528) | Device ID (0–24; 25 different Android devices) |
| Timestamp (Attr 529) | UNIX timestamp of capture |

### Data Collection
- Created in **2013**
- **20+ different users** with varying heights (158–186 cm)
- **25 Android devices** with different models and Android versions (2.3.4 to 4.3)
- Device heterogeneity is a key feature — different WiFi chipsets produce different RSSI readings for the same location

### Key Dataset Properties
- **No missing values** (per UCI documentation)
- Dataset characteristics: Multivariate
- Feature type: Integer, Real
- Associated tasks: **Classification** (building, floor) and **Regression** (longitude, latitude)

---

## Baseline Results Reported

The paper established baseline results using **1-Nearest Neighbor (1-NN) with Euclidean distance** as the fingerprint matching algorithm.

> **[NEEDS VERIFICATION]:** The exact baseline accuracy numbers (building hit rate, floor hit rate, mean positioning error in meters) should be confirmed by reading the full paper text. The UCI page and search results confirm the methodology but I have not directly read the full paper tables.

---

## Relevance to Our Project

This is the **core dataset** for our project. Key connections:

1. **Classification targets:** BuildingID (3 classes) and Floor (5 classes) — our classification tasks
2. **Regression targets:** Longitude and Latitude — our regression tasks
3. **Input features:** 520 WAP RSSI values — these are the raw features we will preprocess
4. **Multi-device, multi-user data:** Makes the problem realistic but also introduces device heterogeneity challenges
5. **The special value +100:** Requires careful preprocessing — represents "not detected" and must be handled appropriately (not treated as a high signal strength)

### Preprocessing Considerations from the Dataset
- The +100 value for undetected WAPs needs to be replaced (common choices: −110, −105, or 0 after normalization)
- Many WAPs may have very few non-null readings → feature selection or filtering needed
- Longitude/Latitude are in a **local coordinate system**, not standard geographic coordinates
- Train/validation split is predefined by the authors (different users for validation)

---

## Questions for Further Investigation

1. What were the exact numerical baseline results (building accuracy, floor accuracy, positioning error)?
2. How did the authors handle the +100 "not detected" value in their baseline experiments?
3. Are there known issues with the dataset (label noise, duplicate fingerprints, etc.) documented in follow-up work?
4. What distance metric performed best in the follow-up paper by Torres-Sospedra et al. (2015)?

---

## Key Quotes to Find

> [TODO: Extract key quotes when accessing the full paper text]

---

*Last updated: 2026-10-21*
*Status: Core facts verified via UCI repository and DOI confirmation. Baseline result numbers need direct paper verification.*
