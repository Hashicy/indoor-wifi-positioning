# Indoor Wi-Fi Positioning System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: UJIIndoorLoc](https://img.shields.io/badge/Benchmark-UJIIndoorLoc-informational.svg)](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
[![ML Framework: scikit-learn & PyTorch](https://img.shields.io/badge/ML-scikit--learn%20%7C%20PyTorch-orange.svg)](https://scikit-learn.org/)
[![Report: IEEEtran](https://img.shields.io/badge/Report-IEEEtran%20LaTeX-red.svg)](report/report.pdf)
[![Presentation: Slide Deck](https://img.shields.io/badge/Presentation-PDF%20Slides-brightgreen.svg)](presentation/presentation.pdf)

An end-to-end machine learning system for indoor positioning using Received Signal Strength Indicator (RSSI) fingerprinting on the **UJIIndoorLoc** campus benchmark. The system formulates indoor localization as a simultaneous hierarchical classification and metric regression problem, predicting campus building, vertical floor level, and continuous metric UTM coordinates from ambient Wi-Fi scans.

---

## Table of Contents
- [Executive Summary & Highlights](#executive-summary--highlights)
- [The Problem & Challenges](#the-problem--challenges)
- [Benchmark Dataset (UJIIndoorLoc)](#benchmark-dataset-ujiindoorloc)
- [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
  - [The Missing-Signal Sentinel Problem ($+100 \to -110\text{ dBm}$)](#1-the-missing-signal-sentinel-problem-100-to--110text-dbm)
  - [Binary Missingness Indicators](#2-binary-missingness-indicators)
  - [Scan-Level Aggregate Features](#3-scan-level-aggregate-features)
  - [Ablation Study Results](#ablation-study-results)
- [Machine Learning Methodology](#machine-learning-methodology)
  - [Classical Baselines](#classical-baselines)
  - [Why Coordinate Regression Over Classification](#why-coordinate-regression-over-classification)
  - [Deep Multitask Learning (Caruana Inductive Transfer)](#deep-multitask-learning-caruana-inductive-transfer)
- [Experimental Results & Analysis](#experimental-results--analysis)
  - [Master Benchmark on Held-Out Test Split](#master-benchmark-on-held-out-test-split)
  - [Building-by-Building Performance Breakdown](#building-by-building-performance-breakdown)
  - [Failure Modes & Physical Error Analysis](#failure-modes--physical-error-analysis)
  - [Out-of-Distribution Robustness: User Grouped CV](#out-of-distribution-robustness-user-grouped-cv)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
- [Deliverables & Academic Artifacts](#deliverables--academic-artifacts)
- [Citation & References](#citation--references)

---

## Executive Summary & Highlights

- **Multi-Target Formulation:** Simultaneously predicts **Building** (3 classes), **Floor** (5 levels), and **2D UTM Coordinates** (continuous longitude/latitude in meters).
- **Leakage-Free Feature Engineering:** Converts 520 raw, sentinel-distorted access point columns into a structured **935-dimensional representation** (465 continuous RSSI + 465 binary missingness flags + 5 scan aggregates) with strict training-only fitting.
- **Sentinel Restoration:** Solved the $+100\text{ dBm}$ unobserved signal flaw by mapping to $-110\text{ dBm}$ (sub-noise floor), boosting floor classification accuracy by **$+3.69\%$** and cutting coordinate error by **$1.53\text{ m}$**.
- **Champion Model Performance:** On the official held-out validation set ($N=1{,}111$ scans, 9 unseen phone models, 4-month temporal drift):
  - **Random Forest ($B=200$):** **99.91% Building Accuracy**, **90.19% Floor Accuracy**, **12.28 m Mean Positioning Error** (and **8.00 m Median Error**).
  - **Extra Trees Regressor:** Achieved **7.70 m Median Error** and **27.00 m 90th-percentile error**.
  - **Building 0 Sub-Domain:** Reached **100.0% Building Accuracy**, **96.64% Floor Accuracy**, and **8.60 m Mean Error** (**5.33 m Median Error**).

---

## The Problem & Challenges

In modern indoor environments, Global Navigation Satellite Systems (GNSS / GPS) are unusable:
1. **RF Attenuation:** Microwave signals at 1.5 GHz suffer massive power loss through concrete slabs, low-E glass, and structural steel.
2. **Multipath Distortion:** Signals bounce off walls, furniture, and partitions, destroying geometric time-of-flight trilateration.
3. **No Extra Hardware Deployment:** While Ultra-Wideband (UWB) offers centimeter accuracy, it requires expensive dedicated infrastructure. Wi-Fi RSSI fingerprinting relies entirely on existing 802.11 access points.

### Three Inherent ML Challenges:
- **Extreme Sparsity:** Over $96.5\%$ of values in any scan are unobserved (a typical device sees only 15–18 out of 520 campus APs).
- **Device Heterogeneity:** Different smartphone antenna gains introduce systematic $10\text{–}15\text{ dBm}$ calibration offsets.
- **Coupled Discrete & Continuous Space:** Predicting continuous coordinates alone is dangerous; mispredicting the floor by 3 vertical meters can lead to severe operational failures in emergency response.

---

## Benchmark Dataset (UJIIndoorLoc)

The system is evaluated on the standard global benchmark **[UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)**:
- **Coverage:** $\approx 110{,}000\text{ m}^2$ across 3 multi-story university buildings.
- **Topology:** 13 valid building–floor combinations (Floor 4 exists only in Building 2).
- **Training Set ($N = 19{,}937$ scans):** 18 users, 16 phone models, collected May–June 2013.
- **Held-Out Validation Set ($N = 1{,}111$ scans):** Single anonymized user, 11 phone models (**9 unseen in training**), collected September–October 2013 (**4-month temporal shift**).

```
Campus Layout (UTM Projected Coordinates):
┌────────────────────────────────────────────────────────┐
│   Building 0                  Building 1               │
│   (4 Floors, 9,116 scans)    (4 Floors, 5,196 scans)   │
│   Dense office partitions    Open central atriums      │
│                                                        │
│               Building 2                               │
│               (5 Floors, 5,625 scans)                  │
│               Modern open layout                       │
└────────────────────────────────────────────────────────┘
```

---

## Feature Engineering & Preprocessing

The preprocessing module [`src/preprocessing.py`](src/preprocessing.py) converts raw scans into an engineered, leakage-free representation.

```
Raw Scans (520 WAPs, +100 sentinels)
  │
  ├──► [Data Cleaning] ─────────► Drop 637 duplicates & 76 empty scans
  ├──► [Zero-Variance Pruning] ─► Drop 55 dead APs (520 ──► 465 active WAPs)
  ├──► [Sentinel Imputation] ──► +100 ──► -110 dBm (preserves physical monotonicity)
  ├──► [Missingness Indicators]► 465 binary flags (1 if unobserved, 0 if detected)
  ├──► [Scan Aggregates] ──────► 5 context features (num_aps, max, mean, std, range)
  └──► [Train-Fitted Scaler] ──► StandardScaler fitted strictly on training data
                                 ────────────────────────────────────────────────
                                 Final Feature Vector Dimension: 935 Features
```

### 1. The Missing-Signal Sentinel Problem ($+100 \to -110\text{ dBm}$)
- **The Issue:** UJIIndoorLoc encodes undetected access points as $+100$. Genuine RSSI values range from $-104\text{ dBm}$ to $0\text{ dBm}$. Treating $+100$ as numeric energy misleads distance metrics ($k$-NN calculates $\Delta = 140$) and destroys physical monotonicity in decision trees.
- **The Fix:** Impute $+100 \to -110\text{ dBm}$, which is strictly below the hardware sensitivity limit ($-104\text{ dBm}$), establishing the natural relationship:
  $$\text{Strong } (-30\text{ dBm}) > \text{Weak } (-85\text{ dBm}) > \text{Noise Floor } (-104\text{ dBm}) > \text{Undetected } (-110\text{ dBm})$$

### 2. Binary Missingness Indicators
In indoor radio environments, signal absence is informative physical evidence of thick concrete obstruction. We construct binary flags $\text{WAP}_k\text{\_miss} \in \{0, 1\}$. 
- **Impact:** Feature importance shows missingness flags account for **$>35\%$ of all node split decisions** in Random Forest.

### 3. Scan-Level Aggregate Features
Derived from detected APs within each scan:
- `num_aps`: Density of local RF coverage (differentiates Building 2's median 19 APs from Buildings 0 & 1's 15 APs).
- `max_rssi`: Proximity to the nearest line-of-sight transmitter.
- `mean_rssi`, `std_rssi`, `range_rssi`: Local RF attenuation and dispersion context.

### Ablation Study Results
Systematic ablation benchmark on the official held-out test set (Random Forest, 100 trees):

| Configuration | Feature Dim | Building Acc | Floor Acc | Mean Error | Median Error |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Raw ($+100$ sentinel)** | 465 | 100.00% | 87.58% | 13.83 m | 9.51 m |
| **2. Imputed ($-110$ dBm)** | 465 | 99.91% | **91.27%** | **12.30 m** | 8.17 m |
| **3. Imputed + Indicators** | 930 | 99.91% | 90.28% | 12.30 m | 8.08 m |
| **4. Imputed + Ind + Aggregates** | 935 | 99.82% | 89.74% | 12.88 m | 8.00 m |
| **5. Full (Standard Scaled)** | 935 | 99.82% | 89.74% | 12.81 m | **7.99 m** |

> **Key Finding:** Simply fixing the sentinel value ($+100 \to -110\text{ dBm}$) produces an immediate **$+3.69\%$ jump in floor accuracy** and cuts positioning error by **$1.53\text{ m}$**. Standard scaling is invariant for tree splits but mandatory for distance and gradient-based models.

---

## Machine Learning Methodology

### Classical Baselines
We benchmark four model families representing distinct inductive biases:
1. **Random Forest ($B=200$ trees):** De-correlated bagging with random subspace projection ($m \approx \sqrt{p}$). Ensemble variance contracts via:
   $$\text{Var}(\bar{f}(\mathbf{x})) = \rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$
2. **Extra Trees Regressor:** Draws cut-point thresholds uniformly at random within feature ranges, smoothing piecewise-constant regression plateaus.
3. **Distance-Weighted $k$-NN ($k=5$):** Inverse-distance weighted neighborhood averaging in Euclidean signal space.
4. **XGBoost (GBDT):** Regularized gradient boosted decision trees with second-order Taylor expansion loss minimization.
5. **Dummy Baseline:** Theoretical minimum predicting most frequent building/floor and median coordinates.

### Why Coordinate Regression Over Classification
Instead of discretizing buildings into discrete grid cells, metric coordinates $(x, y) \in \mathbb{R}^2$ are modeled via continuous bivariate regression:
- **Continuous Euclidean Topology:** Metric space is smooth. Grid classification destroys proximity: misclassifying a point into an adjacent cell ($0.5\text{ m}$ away) receives the identical cross-entropy penalty as predicting a cell $300\text{ m}$ away.
- **Direct Loss Alignment:** Mean Squared Error (MSE) directly minimizes expected squared Euclidean distance:
  $$\mathcal{L}_{\text{coord}} = \frac{1}{N}\sum_{i=1}^N \left[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2\right]$$

### Deep Multitask Learning (Caruana Inductive Transfer)
Based on Rich Caruana’s Multitask Learning theory (1997), a shared deep encoder learns joint topological representations:
- **Shared Backbone:** `Input(935)` $\to$ `Dense(512)` $\to$ `BatchNorm` $\to$ `Dropout(0.3)` $\to$ `Dense(256)` $\to$ `Dense(128)`.
- **Three Task Heads:**
  - Building Head: `Dense(64)` $\to$ `Softmax(3)` (Categorical Cross-Entropy)
  - Floor Head: `Dense(64)` $\to$ `Softmax(5)` (Categorical Cross-Entropy)
  - Coordinate Head: `Dense(64)` $\to$ `Linear(2)` (MSE Loss)
- **Harmonized Loss:** Coordinates are normalized to $[0, 1]$ during backpropagation to prevent million-scale UTM metric losses from swamping classification gradients.

---

## Experimental Results & Analysis

### Master Benchmark on Held-Out Test Split
Evaluated on the official validation set ($N = 1{,}111$, 9 unseen phone models, 4-month temporal gap):

| Model | Building Acc | Building F1 | Floor Acc | Floor F1 | Mean Error (m) | Median Error (m) | P90 Error (m) | Coordinate $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest ($B=200$)** | **99.91%** | **0.9991** | **90.19%** | 0.8801 | **12.28 m** | 8.00 m | 29.40 m | **0.9791** |
| **Extra Trees Regressor** | **99.91%** | **0.9991** | **90.19%** | 0.8801 | 12.63 m | **7.70 m** | **27.00 m** | 0.9729 |
| **$k$-NN ($k=5$, dist-weighted)** | 98.65% | 0.9847 | 75.79% | 0.7493 | 14.26 m | 8.11 m | 27.98 m | 0.9532 |
| **XGBoost (GBDT)** | 99.82% | 0.9983 | 89.11% | **0.8929** | 22.75 m | 15.08 m | 49.10 m | 0.9425 |
| **Dummy Baseline (Prior)** | 24.12% | 0.1296 | 15.48% | 0.0536 | 161.37 m | 142.88 m | 263.79 m | -0.6412 |

---

### Building-by-Building Performance Breakdown

Validation scans are distributed across buildings differently from training data (Building 0 has 48.2% of validation scans vs. 26.4% in training):

| Model | Building | Samples | Building Acc | Floor Acc | Mean Error (m) | Median Error (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **Building 0** | 536 | **100.00%** | **96.64%** | **8.60 m** | **5.33 m** |
| Random Forest | Building 1 | 307 | 99.67% | 80.13% | 14.77 m | 10.05 m |
| Random Forest | Building 2 | 268 | 100.00% | 88.81% | 16.77 m | 11.51 m |
| **Extra Trees** | **Building 0** | 536 | **100.00%** | **96.64%** | **8.47 m** | **5.42 m** |
| Extra Trees | Building 1 | 307 | 99.67% | 80.13% | 16.10 m | 9.73 m |
| Extra Trees | Building 2 | 268 | 100.00% | 88.81% | 16.99 m | 11.83 m |
| **$k$-NN ($k=5$)** | Building 0 | 536 | 98.88% | 76.31% | 12.94 m | 6.73 m |
| $k$-NN ($k=5$) | Building 1 | 307 | 98.70% | 71.99% | 12.84 m | 8.75 m |
| $k$-NN ($k=5$) | Building 2 | 268 | 98.13% | 79.10% | 18.54 m | 9.96 m |

---

### Failure Modes & Physical Error Analysis

1. **Why Building 1 Has Lower Floor Accuracy ($80.13\%$ vs $96.64\%$):**
   - Architectural blueprints reveal Building 1 features large **central open atriums**.
   - Radio waves propagate vertically across mezzanine balconies without concrete floor slab attenuation, leading to adjacent floor confusion.
2. **Floor Errors are Strictly Adjacent ($\pm 1$ Level):**
   - Over **95% of floor errors** occur between immediately adjacent floors (e.g., Floor 1 predicted as Floor 2 near stairwells or elevator shafts). Zero errors span $\pm 3$ or $\pm 4$ floors.
3. **Why $k$-NN Collapsed on the Held-Out Test Set ($97.2\% \to 75.79\%$):**
   - 9 validation phone models were unseen during training. Antenna sensitivity differences cause systematic $10\text{–}15\text{ dBm}$ gain offsets.
   - Uniform shifts warp $L_2$ signal distances, degrading nearest-neighbor retrieval, while decision trees remain robust to monotonic shifts.
4. **Spatial Localization Hotspots:**
   - Errors peak at building perimeters and outdoor balconies due to geometric dilution of precision (APs visible from only one side) and high multipath reflection.

---

### Out-of-Distribution Robustness: User Grouped CV

To evaluate sensitivity to human body attenuation, we ran a 5-fold `GroupKFold` grouped by `USERID` across the 18 training users:
- **Building Accuracy:** Remained rock-solid at **$98.72\%$**.
- **Floor Accuracy:** Dropped to **$63.72\%$**.
- **Mean Coordinate Error:** Increased from $4.5\text{ m}$ to **$16.73\text{ m}$**.
- **Physical Cause:** Human bodies (60% water) absorb 2.4 GHz radio waves. User posture, phone grip, and walking direction introduce body shadowing of up to $10\text{ dBm}$, highlighting human-dependent calibration as a primary domain shift.

---

## Repository Structure

```
indoor-wifi-positioning/
├── data/                               # Dataset documentation and raw files
│   ├── README.md                       # UJIIndoorLoc dataset documentation
│   ├── trainingData.csv                # 19,937 training scans (529 columns)
│   └── validationData.csv              # 1,111 validation scans (held-out test set)
├── notebooks/                          # Interactive exploratory & modeling notebooks
│   ├── 01_dataset_eda.ipynb            # Exploratory data analysis & device shift audit
│   ├── 02_preprocessing_features.ipynb # Preprocessing pipeline & ablation study
│   ├── 03_classical_ml_models.ipynb    # KNN, RF, Extra Trees, XGBoost training
│   ├── 04_evaluation_visualization.ipynb # Master evaluation, CDFs & spatial error maps
│   └── 05_multitask_deep_learning.ipynb# Multitask neural network experiments
├── presentation/                       # Presentation deliverables
│   └── presentation.pdf                # Compiled slide deck for project defense
├── report/                             # Academic paper (IEEEtran format)
│   ├── main.tex                        # Master LaTeX document
│   ├── references.bib                  # BibTeX bibliography
│   ├── report.pdf                      # Compiled academic report
│   ├── sections/                       # Modular LaTeX sections
│   │   ├── dataset_eda.tex             # EDA & data audit
│   │   ├── feature_engineering.tex     # Preprocessing & ablation study
│   │   ├── introduction.tex            # Problem statement & motivation
│   │   ├── related_work.tex            # Literature review & matrix
│   │   └── traditional_ml.tex          # Model formulations & comparative evaluation
│   └── model_evaluation/               # Consolidated CSV performance summaries
├── research/                           # Theoretical research & viva preparation
│   ├── literature_review_draft.md      # Literature survey across 10 core papers
│   ├── literature_review_matrix.csv    # Comparative taxonomy of indoor localization
│   ├── problem_formulation.md          # Formal mathematical formulation
│   ├── research_gap.md                 # Caruana inductive transfer hypotheses
│   └── viva_questions.md               # 20 Viva Voce questions & model answers
├── results/                            # Output metrics, tables, and visual figures
│   ├── figures/                        # High-resolution publication plots
│   │   ├── classical_confusion_matrices.png
│   │   ├── classical_error_boxplots.png
│   │   ├── classical_feature_importance.png
│   │   ├── classical_hyperparameter_tuning.png
│   │   ├── classical_spatial_errors.png
│   │   ├── eda_building_floor_classes.png
│   │   ├── eda_phone_differences.png
│   │   ├── eval_error_cdf.png
│   │   ├── eval_spatial_error_all_models.png
│   │   └── preproc_ablation_comparison.png
│   └── metrics/                        # Raw CSV benchmark results
│       ├── ablation_results.csv
│       ├── classical_ml_summary.csv
│       ├── eval_building_breakdown.csv
│       ├── eval_floor_breakdown.csv
│       ├── eval_master_comparison.csv
│       └── user_grouped_cv_results.csv
├── scripts/                            # Standalone execution scripts
│   └── run_ablation.py                 # Automated ablation benchmark runner
├── src/                                # Core production library
│   ├── __init__.py
│   ├── data_loader.py                  # Loading, cleaning & train/val splitting
│   ├── preprocessing.py               # WifiPreprocessor scikit-learn transformer
│   └── evaluate.py                     # Master evaluation & metric generation
└── requirements.txt                    # Project dependencies
```

---

## Quickstart & Reproduction Guide

### 1. Environment Setup
Clone the repository and install dependencies in a clean virtual environment:
```bash
git clone https://github.com/Hashicy/indoor-wifi-positioning.git
cd indoor-wifi-positioning

# Create and activate environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Preprocessing Ablation Study
To verify the impact of sentinel imputation and engineered features:
```bash
python scripts/run_ablation.py
```
*Outputs are saved to `results/metrics/ablation_results.csv` and `results/figures/preproc_ablation_comparison.png`.*

### 3. Run Master Model Evaluation
To reproduce the full benchmark on the held-out validation set across all models:
```bash
python src/evaluate.py
```
*Outputs detailed metrics to `results/metrics/` and spatial error visualizations to `results/figures/`.*

### 4. Interactive Jupyter Notebooks
Launch the notebook environment to explore individual modules:
```bash
jupyter notebook notebooks/
```
- Start with `01_dataset_eda.ipynb` for dataset distributions and device shift analysis.
- Follow with `02_preprocessing_features.ipynb` and `03_classical_ml_models.ipynb`.
- Review `04_evaluation_visualization.ipynb` for publication-ready error CDFs and spatial heatmaps.

---

## Deliverables & Academic Artifacts

- **Compiled Research Paper (PDF):** [`report/report.pdf`](report/report.pdf)
  - 8-page IEEEtran conference format including literature review, methodology, mathematical formulations, and detailed error analyses.
- **Presentation Slide Deck (PDF):** [`presentation/presentation.pdf`](presentation/presentation.pdf)
  - Comprehensive 22-slide presentation deck covering motivation, data engineering, ablation studies, master results, and defense answers.
- **Viva Voce Defense Guide:** [`research/viva_questions.md`](research/viva_questions.md)
  - 20 defense questions with technical answers covering antenna gains, data leakage, and loss weighting.

---

## Citation & References

1. **UJIIndoorLoc Benchmark:**
   > J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building and multi-floor database for WLAN fingerprint-based indoor localization problems," *International Conference on Indoor Positioning and Indoor Navigation (IPIN)*, 2014.
2. **Multitask Learning Foundation:**
   > R. Caruana, "Multitask Learning," *Machine Learning*, vol. 28, no. 1, pp. 41–75, 1997.
3. **Random Forests:**
   > L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
4. **Distance Measures for Wi-Fi Fingerprinting:**
   > J. Torres-Sospedra et al., "Comprehensive analysis of distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems," *Expert Systems with Applications*, vol. 42, no. 23, 2015.
