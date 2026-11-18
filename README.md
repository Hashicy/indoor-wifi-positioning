# Indoor Wi-Fi Positioning System
## Signal Strength Features, Deep Multitask Learning & Hybrid Ensembles

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: UJIIndoorLoc](https://img.shields.io/badge/Benchmark-UJIIndoorLoc-informational.svg)](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
[![Deep Learning: PyTorch](https://img.shields.io/badge/DL-PyTorch%202.15-red.svg)](https://pytorch.org/)
[![ML Framework: scikit-learn](https://img.shields.io/badge/ML-scikit--learn%201.9-orange.svg)](https://scikit-learn.org/)
[![Tests: 63](https://img.shields.io/badge/Unit%20Tests-63-brightgreen.svg)](tests/)
[![Report: IEEEtran](https://img.shields.io/badge/Report-IEEEtran%20LaTeX-blueviolet.svg)](report/main.tex)
[![Presentation: PDF](https://img.shields.io/badge/Presentation-Phase%202%20PDF-success.svg)](presentation/Phase2_Presentation_9pages.pdf)

An end-to-end deep learning and machine learning system for indoor positioning using Received Signal Strength Indicator (RSSI) fingerprints on the **UJIIndoorLoc** multi-building campus benchmark. The project treats indoor localization as a joint classification and regression task: from a single Wi-Fi scan it predicts the campus **building**, the **floor**, and continuous **(x, y) coordinates** in UTM metres.

**Phase 1** built a leakage-safe preprocessing pipeline and classical baselines (k-NN, Random Forest, Extra Trees, XGBoost). **Phase 2** adds a deep multitask neural network, hybrid deep learning + tree models, loss-balancing and regularization ablations, and a full spatial error analysis.

---


## Team & Workstream Responsibilities

| Student Name | Enrollment No. | Email |
| :--- | :--- | :--- |
| Harshita Joshi | 2401010184 | harshita.joshi2024@nst.rishihood.edu.in |
| Soumya Tiwari | 2401020111 | soumya.tiwari2024@nst.rishihood.edu.in |
| Niharika Choudhary | 2401010298 | niharika.choudhary2024@nst.rishihood.edu.in |
| Shreshtha Gupta | 2401010446 | shreshtha.gupta2024@nst.rishihood.edu.in |
| Karan Chhillar | 2401010211 | karan.chhillar2024@nst.rishihood.edu.in |

The Phase 2 work was split into five workstreams:
- **Member 1 – Deep Learning Architecture & Core Pipeline:** dataset inspection, position-grouped data split (`GroupShuffleSplit`), preprocessing, the 512–256–128 shared-trunk multitask network, and single-task baselines.
- **Member 2 – Hybrid Deep Learning + Machine Learning:** classical baseline audit, out-of-fold (OOF) tree probability stacking, and gradient-boosted (GBDT) residual coordinate correction.
- **Member 3 – Loss Functions, Regularization & Ablation:** homoscedastic uncertainty weighting (Kendall et al.), dropout, weight decay, RSSI Gaussian noise, and access-point dropout.
- **Member 4 – Evaluation, Error Analysis & Visualization:** metrics, EvAAL scoring, confusion matrices, error histograms, CDFs, spatial error maps, and failure-case analysis.
- **Member 5 – Literature, Integration & Reporting:** literature review, theory, unit-test consolidation, the IEEE report (`report/main.tex`), the presentation, and viva preparation.

---

## Executive Summary & Master Benchmark

All models are evaluated on the **same held-out test set**: the 1,111 scans of the official `validationData.csv`, recorded about **four months after** the training data, by 11 phones of which **9 never appear in training**. The test set is used once, for final results only; model selection uses a separate position-grouped validation split.

| Model | Paradigm | Building Acc | Floor Acc | Joint Acc | Mean Error (m) | Median Error (m) | P90 Error (m) | Max Error (m) | R² | EvAAL Score ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random Forest (Phase 1 baseline) | Classical ML | 99.73% | 89.92% | 89.74% | 12.30 | 8.15 | 29.55 | 98.42 | 0.978 | 13.04 |
| Extra Trees | Classical ML | 99.73% | 89.92% | 89.74% | 12.46 | 7.66 | 25.68 | 94.10 | 0.971 | 13.20 |
| k-NN (k = 5, distance-weighted) | Classical ML | 98.92% | 75.43% | 75.16% | 13.68 | 8.20 | 29.21 | 115.30 | 0.964 | 15.54 |
| XGBoost | Classical ML | 99.82% | 89.56% | 89.56% | 22.70 | 15.17 | 47.77 | 135.20 | 0.940 | 23.41 |
| Single-task deep networks | Deep Learning | **100.00%** | 91.90% | 91.90% | 8.66 | 6.37 | **17.13** | 81.35 | 0.984 | 9.02 |
| Multitask neural network | Deep Learning | 99.73% | 91.00% | 91.00% | 8.95 | 7.04 | 17.87 | 80.72 | 0.982 | 9.50 |
| Hybrid A: tree probability stacking | Hybrid DL+ML | 99.82% | **92.08%** | **92.08%** | 8.86 | 6.69 | 17.72 | 80.12 | 0.983 | 9.31 |
| Hybrid B: residual GBDT correction | Hybrid DL+ML | 99.73% | 91.00% | 91.00% | 8.75 | 6.69 | 17.59 | 79.45 | 0.983 | 9.29 |
| **Full Hybrid (stacking + residual GBDT)** | **Hybrid DL+ML** | 99.82% | **92.08%** | **92.08%** | **8.51** | **6.30** | 17.58 | **78.90** | **0.984** | **8.96** |

*Source: [`outputs/final_comparison.csv`](outputs/final_comparison.csv). Deep learning rows use seed 42; the EvAAL score is the IPIN 2015 competition metric: mean error + 50 m per wrong building + 4 m per floor of error (lower is better).*

**Headline:** the Full Hybrid reaches **8.51 m mean error** (−30.8% vs Random Forest), **6.30 m median error**, **17.58 m P90 error** (−40.5%) and **92.08% floor accuracy** (+2.16 points), with the best EvAAL score (**8.96**). Single-task networks are a close second (EvAAL 9.02) and remain best on building accuracy and P90 error.

---

## Key Findings

1. **Neural networks beat tree models on coordinates.** Moving from Random Forest to a neural network cuts mean error by 27–30% and P90 error by 40–43% (3-seed averages): smooth regression avoids the piecewise-constant output of decision trees.
2. **The Full Hybrid is the best overall system.** Tree probability stacking mainly fixes floors (+1.08 points, 100 → 88 floor errors); residual boosting mainly sharpens coordinates (median −0.74 m). The Full Hybrid is also best on validation (6.33 m vs 6.45 m mean error), so the choice is not driven by the test set.
3. **Sharing alone is not a free win.** With an unweighted loss, the multitask network matches but does not beat single-task networks (mean ± std over seeds 42, 7, 123):

   | Test metric | Multitask | Single-task | Phase 1 RF |
   | :--- | :---: | :---: | :---: |
   | Building accuracy (%) | 99.88 ± 0.14 | 99.97 ± 0.05 | 99.73 |
   | Floor accuracy (%) | 91.21 ± 0.23 | 91.51 ± 0.68 | 89.92 |
   | Mean error (m) | 9.03 ± 0.10 | 8.59 ± 0.11 | 12.30 |
   | Median error (m) | 7.17 ± 0.14 | 6.30 ± 0.13 | 8.15 |
   | P90 error (m) | 17.70 ± 0.27 | 16.91 ± 0.25 | 29.55 |

   A likely cause is loss imbalance: the floor cross-entropy is about 20× larger than the coordinate loss, so it dominates the shared trunk.
4. **Simple regularization beats clever weighting.** Dropout clearly matters; uncertainty weighting and weight decay made results worse.
5. **Remaining errors have physical causes.** 90 of 100 floor errors are off by exactly one floor, and three of the four largest coordinate errors (75–81 m) come from scans with only 6, 1 and 4 visible access points.

---

## System Architecture & Methods

```
RSSI scan (459 APs) ─► cleaning & grouped split ─► ┬─► multitask neural net ─┬─► hybrid DL + ML ─► building · floor · (x, y)
                                                   └─► classical trees ──────┘
```

### 1. Data Preprocessing & Leakage-Free Split
- **Cleaning:** 637 duplicate scans and 73 empty scans are removed from `trainingData.csv` (19,937 → 19,227 scans).
- **Position-grouped split (`GroupShuffleSplit`, seed 42):** 15% of reference positions form the validation set. Each position was scanned about 20 times, so a random split would put near-identical scans in both sets. Result: **16,432 train / 2,795 validation** scans with no shared position; all 13 building–floor classes appear in both.
- **Test set:** the official `validationData.csv` (1,111 scans), used only for final results.
- **Sentinel handling (+100 → −110 dBm):** undetected access points are mapped below the weakest real reading (−104 dBm), then min–max scaled to [0, 1], so absent APs map to 0 and signal order is preserved.
- **Features:** the 459 access points observed in the training part. Scalers and label encoders are **fitted on training data only**.
- **Coordinates:** UTM coordinates (≈ 4.86 × 10⁶ m) are standardized for training and converted back to metres before any error is computed.

### 2. Deep Multitask Neural Network
A hard-parameter-sharing network (Caruana, 1997) with about 4.0 × 10⁵ trainable parameters:
- **Shared trunk:** Linear(459→512) → LayerNorm → GELU → Dropout(0.2) → Linear(512→256) → LayerNorm → GELU → Dropout(0.2) → Linear(256→128) → LayerNorm → GELU.
- **Heads:** building Linear(128→3) with cross-entropy; floor Linear(128→5) with cross-entropy; coordinates Linear(128→2) with MSE (SmoothL1 optional).
- **Loss:** L = L_building + L_floor + L_coords.
- **Training:** Adam, learning rate 10⁻³, batch 128, at most 150 epochs, early stopping after 20 epochs without improvement.
- **Model selection:** the epoch with the best **validation EvAAL score**. Validation loss was misleading: floor cross-entropy rises after a few epochs (over-confidence on hard scans) while accuracy and coordinate error keep improving.
- **Single-task baselines:** the same trunk with one head each, trained with the same split, optimizer and stopping rule.

### 3. Hybrid Stacking & Residual Correction
- **A · Probability stacking:** 100-tree Random Forests produce building and floor class probabilities (3 + 5 = 8 features), computed **out-of-fold** with 5-fold cross-validation so no tree scores a scan it was trained on. The network receives a 459 + 8 = **467-dimensional** input. Full-data forests are refit once and used only for validation and test scans.
- **B · Residual correction:** histogram GBDT regressors (depth 6, L2 λ = 2) learn the network's systematic coordinate residuals, e = y − ŷ_DL, and add them back at inference: ŷ = ŷ_DL + ê. They are fitted on training data only.
- **Full Hybrid:** stacked network + residual correction.

| Experiment (test) | Building | Floor | Mean (m) | Median (m) | P90 (m) | RMSE (m) | EvAAL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DL baseline | 99.73% | 91.00% | 8.95 | 7.04 | 17.87 | 8.58 | 9.50 |
| DL + tree stacking | 99.82% | 92.08% | 8.86 | 6.69 | 17.72 | 8.30 | 9.31 |
| DL + residual GBDT | 99.73% | 91.00% | 8.75 | 6.69 | 17.59 | 8.50 | 9.29 |
| **Full Hybrid** | **99.82%** | **92.08%** | **8.51** | **6.30** | **17.58** | **8.16** | **8.96** |

### 4. Loss Balancing & Regularization Ablation
- **Uncertainty weighting (Kendall et al., CVPR 2018):**
  $$\mathcal{L}_{\text{total}} = \sum_{i} \left[ e^{-s_i} \, \mathcal{L}_i + c_i \, s_i \right]$$
  where $s_i = \log \sigma_i^2$ is learnable, $c_i = 1$ for classification and $c_i = 0.5$ for regression.
- **RSSI noise:** Gaussian noise (σ = 0.02 in scaled units, about ±1–2 dBm) added to detected readings only.
- **Access-point dropout (p = 0.10):** visible AP readings are randomly set to the missing-signal sentinel.

One factor is changed at a time; same split, architecture and evaluation for every run (EvAAL score, lower is better):

| Run | Change vs baseline | Val score | Building | Floor | Mean (m) | Test EvAAL |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| A | Baseline: plain loss, dropout 0.2 | **6.55** | 99.73% | 91.00% | 8.95 | 9.50 |
| B | Uncertainty weighting | 7.02 | 99.82% | 91.09% | 9.19 | 9.69 |
| C | No dropout | 7.08 | 99.55% | 91.00% | 9.49 | 10.12 |
| D | Dropout 0.3 | 6.92 | 99.73% | **91.81%** | **8.85** | **9.36** |
| E | RSSI Gaussian noise | 7.07 | 99.91% | 90.91% | 8.97 | 9.42 |
| F | AP dropout | 7.32 | 99.91% | 90.19% | 9.05 | 9.52 |
| G | Uncertainty + noise + AP dropout | 7.21 | 99.73% | 90.55% | 9.01 | 9.55 |
| WD | Weight decay 10⁻⁵ | 7.46 | 99.73% | 91.45% | 9.43 | 9.94 |
| WD | Weight decay 10⁻⁴ | 7.93 | 100.00% | 91.00% | 10.36 | 10.75 |

- **Uncertainty weighting did not help:** the building loss quickly approaches zero, so its learned weight grows to about 9 × 10⁴ and the balancing chases an already-solved task.
- **Dropout matters:** removing it raises the test EvAAL score to 10.12; dropout 0.3 gives the best test score (9.36).
- **Weight decay hurts:** at 10⁻⁴ the best epoch is only 33 and the mean error rises to 10.36 m.
- **Augmentations are neutral** on this test set, and combining them (G) is not additive.
- **Caution:** run A is still best on validation; D's test gain (0.14) is about the size of seed-to-seed variation.

Source: [`results/metrics/ablation_results_member3.csv`](results/metrics/ablation_results_member3.csv).

### 5. Evaluation & Error Analysis
Metrics: building/floor accuracy and macro-F1, joint building–floor accuracy, Euclidean error $d_i = \sqrt{(x_i-\hat{x}_i)^2 + (y_i-\hat{y}_i)^2}$ (mean, median, P75, P90, max), coordinate MAE/RMSE/R², and the EvAAL score.

- **Share of test scans within a radius:**

  | Within | Random Forest | Multitask DL |
  | :--- | :---: | :---: |
  | < 5 m | 33.4% | 34.9% |
  | < 10 m | 59.0% | 70.2% |

  The models are close below 5 m; the network's advantage lies mainly in reducing medium and large errors.
- **Floors:** 100 floor errors (Multitask DL), of which 90 are adjacent-floor confusions. The largest are floor 1 → 2 (38 scans) and 0 → 1 (15). Whenever the floor is right, the building is right too.
- **Buildings:** only 3 building errors in 1,111 scans.
- **Per building (Multitask DL):** Building 0 mean 7.09 m / median 5.76 m; Building 1 mean 10.96 m / median 8.52 m; Building 2 mean 10.38 m / median 7.71 m.
- **Visible access points:** sparse scans (≤ 11 APs) have the highest mean error (10.89 m) and include the maximum error (80.72 m); scans with 11–15 APs are the most accurate (7.01 m).
- **Failure cases:** 178 test scans are flagged (wrong building, wrong floor, or error ≥ P90 = 17.87 m), logged in [`outputs/failure_cases_analysis.csv`](outputs/failure_cases_analysis.csv). Building 1 accounts for 90 of them, and Phone 20 for 54 (30%), pointing to device-dependent RSSI differences.

---

## Repository Structure

```
indoor-wifi-positioning/
├── data/
│   ├── README.md                      # Dataset download & preparation instructions
│   ├── trainingData.csv               # Official UJIIndoorLoc training data (not committed)
│   ├── validationData.csv             # Official held-out test data (not committed)
│   ├── preprocessed_train.csv         # Phase 1 engineered features (train)
│   ├── preprocessed_val.csv           # Phase 1 engineered features (test)
│   ├── preprocessed_dataset.zip       # Compressed copy of the preprocessed data
│   └── processed/                     # Separate X / y / meta CSVs and feature dictionary
├── models/
│   ├── dl_preprocessing.joblib        # Fitted scaler, label encoders and WAP list
│   ├── multitask_best.pth             # Best multitask checkpoint
│   ├── single_{building,floor,coords}_best.pth  # Single-task baselines
│   ├── hybrid_multitask_stacked_best.pth        # Stacked hybrid network
│   └── ablation_{A..G,WD_1e5,WD_1e4}_best.pth   # Ablation run checkpoints
├── notebooks/
│   ├── 01_dataset_eda.ipynb                     # Exploratory data analysis
│   ├── 02_preprocessing_features.ipynb          # Preprocessing & feature engineering
│   ├── 03_classical_ml_models.ipynb             # Classical baselines (k-NN, RF, ET, XGBoost)
│   ├── 04_evaluation_visualization.ipynb        # Classical evaluation & plots
│   ├── 05_multitask_deep_learning.ipynb         # Multitask & single-task networks
│   ├── 06_hybrid_dl_ml.ipynb                    # Stacking & residual correction
│   ├── 07_loss_regularization_ablation.ipynb    # Loss balancing & ablation
│   └── 08_dl_evaluation_visualization.ipynb     # Final evaluation & error analysis
├── outputs/
│   ├── final_comparison.csv           # Master metrics table for all models
│   ├── failure_cases_analysis.csv     # The 178 flagged test scans
│   ├── confusion_matrix_{building,floor}.png
│   ├── error_cdf.png                  # Error CDF across models
│   ├── error_distribution.png         # Error histograms and box plots
│   └── spatial_error_plot.png         # Spatial error map per building
├── presentation/
│   ├── Phase2_Presentation_9pages.pdf # Phase 2 presentation (title + 8 slides)
│   ├── presentation.pdf               # Phase 1 presentation
│   └── final_presentation.pptx        # Earlier 12-slide team deck
├── report/
│   ├── main.tex                       # IEEE conference report (Phase 2)
│   ├── sections/                      # One LaTeX file per report section
│   ├── figures/                       # Figures used by the report
│   ├── model_evaluation/              # Phase 1 evaluation tables (read by notebooks 03–04)
│   ├── references.bib                 # BibTeX bibliography (24 entries)
│   └── report.pdf                     # Compiled Phase 1 report
├── research/
│   ├── literature_review_draft.md     # Literature synthesis
│   ├── literature_review_matrix.csv   # Structured matrix of prior work
│   ├── problem_formulation.md         # Formal problem statement
│   ├── research_gap.md                # Research gaps & objectives
│   ├── paper_notes/                   # Notes on key papers
│   ├── viva_preparation.md            # Answers to the 11 core viva questions
│   └── viva_questions.md              # Extended viva question bank
├── results/
│   ├── figures/                       # EDA, classical, DL, hybrid & ablation plots
│   └── metrics/                       # CSV/JSON metrics, training histories, run info
├── scripts/
│   ├── generate_preprocessed_data.py  # Rebuilds the Phase 1 preprocessed CSVs
│   └── run_ablation.py                # Phase 1 preprocessing ablation
├── src/
│   ├── models/
│   │   ├── multitask_net.py           # Shared-trunk multitask network
│   │   ├── hybrid_model.py            # HybridPositioningSystem wrapper
│   │   └── loss.py                    # Basic combined loss & UncertaintyLoss
│   ├── data_loader.py                 # Raw data loading & column definitions
│   ├── preprocessing.py               # Phase 1 feature pipeline
│   ├── data_preprocessing.py          # DL cleaning, grouped split & scaling
│   ├── dataset.py                     # PyTorch Dataset wrappers
│   ├── augmentations.py               # RSSI noise & AP dropout
│   ├── stacking_features.py           # Out-of-fold tree probability stacking
│   ├── residual_correction.py         # GBDT residual corrector
│   ├── train_multitask.py             # Multitask & single-task training
│   ├── train_hybrid.py                # Hybrid training
│   ├── ablation.py                    # Loss & regularization ablation runner
│   ├── evaluate.py                    # Final evaluation & figure generation
│   └── metrics.py                     # Euclidean error & EvAAL metrics
├── tests/                             # 63 unit tests (7 files)
├── requirements.txt
└── README.md
```

---

## Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
git clone https://github.com/Hashicy/indoor-wifi-positioning.git
cd indoor-wifi-positioning

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Place `trainingData.csv` and `validationData.csv` in `data/` (see [`data/README.md`](data/README.md)).

### 2. Run the Unit Tests
```bash
PYTHONPATH=. pytest
```
The suite has 63 tests covering preprocessing, the network, losses and augmentations, the hybrid pipeline, and the evaluation metrics.

### 3. Preprocess the Data
```bash
PYTHONPATH=. python3 src/data_preprocessing.py
```
Produces `models/dl_preprocessing.joblib` (16,432 train / 2,795 validation / 1,111 test scans; 459 access points).

### 4. Train the Models
```bash
PYTHONPATH=. python3 src/train_multitask.py   # multitask + single-task networks
PYTHONPATH=. python3 src/train_hybrid.py      # stacking & residual GBDT
PYTHONPATH=. python3 src/ablation.py          # loss & regularization ablation
```

### 5. Evaluate and Regenerate Figures
```bash
PYTHONPATH=. python3 src/evaluate.py
```
Writes tables and figures to `outputs/`; copies used in the report live in `report/figures/`.

### 6. Build the Report
```bash
cd report
pdflatex main.tex && pdflatex main.tex
```
Requires a LaTeX distribution with the `IEEEtran` class (included in full TeX Live / MacTeX; with BasicTeX run `sudo tlmgr install ieeetran`). The bibliography is embedded in `main.tex`, so no BibTeX run is needed.

---

## Deliverables

- **IEEE report:** [`report/main.tex`](report/main.tex) — two-column IEEE conference paper covering all five workstreams, with sections in [`report/sections/`](report/sections/).
- **Phase 2 presentation:** [`presentation/Phase2_Presentation_9pages.pdf`](presentation/Phase2_Presentation_9pages.pdf) — title slide plus 8 slides: problem and dataset, pipeline, multitask network, hybrid methods, ablation, final comparison, error analysis, conclusion.
- **Phase 1 presentation:** [`presentation/presentation.pdf`](presentation/presentation.pdf).
- **Viva preparation:** [`research/viva_preparation.md`](research/viva_preparation.md) (11 core questions) and [`research/viva_questions.md`](research/viva_questions.md).
- **Literature review:** [`research/literature_review_matrix.csv`](research/literature_review_matrix.csv) and [`research/literature_review_draft.md`](research/literature_review_draft.md).

---

## Limitations & Future Work

**Limitations**
- Results come from one dataset and one campus, with a fixed set of access points.
- Gaps between the best deep models (0.15–0.44 m mean error) are close to seed-to-seed variation; hybrid results use a single seed.
- Device bias is visible (e.g. Phone 20) but not corrected.

**Future work**
- Hierarchical heads that condition coordinates on the predicted building and floor.
- GradNorm or validation-tuned loss weights instead of uncertainty weighting.
- Per-phone RSSI calibration, and sequence models that use consecutive scans for tracking.
- Multi-seed hybrid runs and cross-dataset evaluation.

---

## Citation & References

```bibtex
@inproceedings{torres2014ujiindoorloc,
  author    = {Torres-Sospedra, Joaqu{\'i}n and Montoliu, Ra{\'u}l and Mart{\'i}nez-Us{\'o}, Adolfo and Avariento, Joan P. and Arnau, Tom{\'a}s J. and Benedito-Bordonau, Mauri and Huerta, Joaqu{\'i}n},
  title     = {{UJIIndoorLoc}: A New Multi-Building and Multi-Floor Database for {WLAN} Fingerprint-Based Indoor Localization Problems},
  booktitle = {Proceedings of the International Conference on Indoor Positioning and Indoor Navigation (IPIN)},
  pages     = {261--270},
  year      = {2014},
  doi       = {10.1109/IPIN.2014.7275492}
}

@article{caruana1997multitask,
  author    = {Caruana, Rich},
  title     = {Multitask Learning},
  journal   = {Machine Learning},
  volume    = {28},
  number    = {1},
  pages     = {41--75},
  year      = {1997},
  doi       = {10.1023/A:1007379606734}
}

@inproceedings{kendall2018multi,
  author    = {Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  title     = {Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {7482--7491},
  year      = {2018},
  doi       = {10.1109/CVPR.2018.00781}
}
```
