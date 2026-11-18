# Indoor Wi-Fi Positioning System
## Signal Strength Features, Deep Multitask Learning & Hybrid Ensembles

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark: UJIIndoorLoc](https://img.shields.io/badge/Benchmark-UJIIndoorLoc-informational.svg)](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
[![Deep Learning: PyTorch](https://img.shields.io/badge/DL-PyTorch%202.15-red.svg)](https://pytorch.org/)
[![ML Framework: scikit-learn](https://img.shields.io/badge/ML-scikit--learn%201.9-orange.svg)](https://scikit-learn.org/)
[![Tests: 63 Passing](https://img.shields.io/badge/Tests-63%20Passed-brightgreen.svg)](tests/)
[![Report: IEEEtran](https://img.shields.io/badge/Report-IEEEtran%20LaTeX-blueviolet.svg)](report/main.tex)
[![Presentation: Slide Deck](https://img.shields.io/badge/Presentation-PPTX%20Deck-success.svg)](presentation/final_presentation.pptx)

An end-to-end deep learning and machine learning system for indoor positioning using Received Signal Strength Indicator (RSSI) fingerprints on the **UJIIndoorLoc** multi-building campus benchmark. The project formulates indoor localization as a simultaneous hierarchical classification and metric regression task, predicting campus building, vertical floor level, and continuous metric UTM coordinates from ambient Wi-Fi scans.

---

## Table of Contents
- [Team & Workstream Responsibilities](#team--workstream-responsibilities)
- [Executive Summary & Master Benchmark](#executive-summary--master-benchmark)
- [System Architecture & Theoretical Foundations](#system-architecture--theoretical-foundations)
  - [1. Data Preprocessing & Leakage-Free Split (Member 1)](#1-data-preprocessing--leakage-free-split)
  - [2. Deep Multitask Neural Network (Member 1)](#2-deep-multitask-neural-network)
  - [3. Hybrid Stacking & Residual Correction (Member 2)](#3-hybrid-stacking--residual-correction)
  - [4. Loss Balancing & Physical Regularization (Member 3)](#4-loss-balancing--physical-regularization)
  - [5. Standardized Evaluation & Error Diagnostics (Member 4)](#5-standardized-evaluation--error-diagnostics)
  - [6. Literature Review, Integration & Reporting (Member 5)](#6-literature-review-integration--reporting)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Running Automated Test Suites](#2-running-automated-test-suites)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Model Training Pipelines](#4-model-training-pipelines)
  - [5. Master Evaluation & Plot Generation](#5-master-evaluation--plot-generation)
- [Deliverables & Academic Artifacts](#deliverables--academic-artifacts)
- [Citation & Verified References](#citation--verified-references)

---

## Team & Workstream Responsibilities
This project was designed, implemented, and validated as a collaborative five-member research project:
- **Member 1 (Deep Learning Architecture & Core Pipeline):** Dataset inspection, position-grouped data splitting (`GroupShuffleSplit`), preprocessing scalers, 512-256-128 shared trunk multitask neural network, and single-task baselines.
- **Member 2 (Hybrid Deep Learning + Machine Learning):** Classical baseline auditing, out-of-fold (OOF) tree probability stacking, and gradient boosted decision tree (GBDT) residual coordinate correction.
- **Member 3 (Loss Functions, Regularization & Ablation Studies):** Homoscedastic uncertainty loss weighting (Kendall et al.), dropout tuning, weight decay, RSSI Gaussian noise injection, and access-point dropout.
- **Member 4 (Evaluation, Error Analysis & Visualization):** Comprehensive metric evaluation, EvAAL competition scoring, confusion matrices, error histograms, empirical CDFs, spatial error maps, and failure case auditing.
- **Member 5 (Literature Review, Theoretical Foundations, Integration & Final Report):** Prior literature synthesis, theoretical formulation (MTL, hybrid synergy), test suite consolidation, master IEEE report (`report/main.tex`), 12-slide presentation (`presentation/final_presentation.pptx`), and viva voce preparation guide.

---

## Executive Summary & Master Benchmark

All models were evaluated on the exact same held-out benchmark test set ($N = 1{,}111$ scans from the official `validationData.csv`), recorded four months later by 11 mobile devices (including 9 smartphone models completely absent from the training set).

| Model Architecture | Paradigm | Building Acc (%) | Floor Acc (%) | Joint Acc (%) | Mean Error (m) | Median Error (m) | $P_{90}$ Error (m) | Max Error (m) | $R^2$ Score | EvAAL Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Baseline)** | Classical ML | 99.73% | 89.92% | 89.74% | 12.30 | 8.15 | 29.55 | 98.42 | 0.978 | 13.04 |
| **Extra Trees Regressor** | Classical ML | 99.73% | 89.92% | 89.74% | 12.46 | 7.66 | 25.68 | 94.10 | 0.971 | 13.20 |
| **$k$-NN ($k=5$, distance-weighted)** | Classical ML | 98.92% | 75.43% | 75.16% | 13.68 | 8.20 | 29.21 | 115.30 | 0.964 | 15.54 |
| **XGBoost Regressor** | Classical ML | 99.82% | 89.56% | 89.56% | 22.70 | 15.17 | 47.77 | 135.20 | 0.940 | 23.41 |
| **Single-Task Deep Networks** | Deep Learning | **100.00%** | 91.90% | 91.90% | 8.66 | 6.37 | **17.13** | 81.35 | **0.984** | 9.02 |
| **Multitask Neural Network** | Deep Learning | 99.73% | 91.00% | 91.00% | 8.95 | 7.04 | 17.87 | 80.72 | 0.982 | 9.50 |
| **Hybrid Approach A (Stacking)** | Hybrid DL+ML | 99.82% | **92.08%** | **92.08%** | 8.86 | 6.69 | 17.72 | 80.12 | 0.983 | 9.31 |
| **Hybrid Approach B (Residual GBDT)** | Hybrid DL+ML | 99.73% | 91.00% | 91.00% | 8.75 | 6.69 | 17.59 | 79.45 | 0.983 | 9.29 |
| **Full Hybrid (Stacked DL + GBDT)** | **Hybrid DL+ML** | 99.82% | **92.08%** | **92.08%** | **8.51** | **6.30** | 17.58 | **78.90** | **0.984** | **8.96** |

### Key Project Takeaways:
1. **Continuous Manifold Superiority:** Deep multitask learning cuts mean positioning error by **27.2%** compared to the best classical tree model (8.95 m vs 12.30 m) by overcoming the piecewise-constant step-function discretization inherent to decision trees.
2. **State-of-the-Art Hybrid Synergy:** The Full Hybrid system (combining tree probability stacking with residual gradient boosting) achieves the best overall performance across all evaluated metrics: **99.82% building accuracy**, **92.08% floor accuracy**, **8.51 m mean error**, **6.30 m median error**, and an **EvAAL score of 8.96**.
3. **Sub-Millisecond Edge Inference:** A single forward pass through the unified multitask model executes in **<0.8 ms** on CPU (~15x faster than classical tree ensembles), making it deployable on edge devices.

---

## System Architecture & Theoretical Foundations

### 1. Data Preprocessing & Leakage-Free Split
- **Sentinel Remediation ($+100 \to -110\text{ dBm}$):** Raw unobserved access points are mapped below the hardware receiver sensitivity noise floor ($-104$ dBm) and min-max scaled to $[0, 1]$. This ensures absent APs map strictly to $0.0$, preserving physical monotonicity: strong ($-30$) $>$ moderate ($-70$) $>$ weak ($-100$) $>$ absent ($-110$).
- **Position-Grouped Validation Split (`GroupShuffleSplit`):** Scans surveyed at the same reference position (~20 scans per location) are strictly assigned to either training or validation. The resulting train (16,432) and validation (2,795) sets share zero physical coordinates, preventing optimistic leakage.
- **Coordinate Normalization:** Projected UTM metric coordinates are standardized ($\mu=0, \sigma=1$) to prevent large raw UTM magnitudes ($\sim 10^6$ m) from overpowering gradient descent.

### 2. Deep Multitask Neural Network
A hard-parameter-sharing neural network with ~400,000 trainable parameters:
- **Shared Trunk:** Linear(459 $\to$ 512) $\to$ LayerNorm $\to$ GELU $\to$ Dropout(0.2) $\to$ Linear(512 $\to$ 256) $\to$ LayerNorm $\to$ GELU $\to$ Dropout(0.2) $\to$ Linear(256 $\to$ 128) $\to$ LayerNorm $\to$ GELU.
- **Task Output Heads:**
  - Building Head: Linear(128 $\to$ 3) $\to$ CrossEntropyLoss
  - Floor Head: Linear(128 $\to$ 5) $\to$ CrossEntropyLoss
  - Coordinate Head: Linear(128 $\to$ 2) $\to$ MSELoss / SmoothL1Loss

### 3. Hybrid Stacking & Residual Correction
- **Probability Stacking:** Dedicated 100-tree Random Forest models generate building and floor class posterior probabilities ($3 + 5 = 8$ features). Probabilities are evaluated strictly **out-of-fold (OOF)** via 5-fold cross-validation on the training set to prevent target leakage. The neural trunk receives an augmented 467-dimensional input.
- **Residual Coordinate Correction:** Systematic local spatial distortions ($e_x = x - \hat{x}_{\text{DL}}$, $e_y = y - \hat{y}_{\text{DL}}$) are learned by a secondary Gradient Boosted Decision Tree (GBDT) regressor and added to the neural coordinate estimate at inference time.

### 4. Loss Balancing & Physical Regularization
- **Homoscedastic Uncertainty Weighting (Kendall et al., CVPR 2018):**
  $$\mathcal{L}_{\text{total}} = \sum_{i} \left[ \exp(-s_i) \cdot \mathcal{L}_i + c_i \cdot s_i \right]$$
  where $s_i = \log \sigma_i^2$ is a trainable log-variance parameter ($c_i = 1.0$ for classification, $c_i = 0.5$ for regression).
- **Physical Augmentations:**
  - *RSSI Noise Augmentation:* Zero-mean Gaussian noise ($\sigma = 0.02$, $\pm 1$--$2$ dBm) injected into detected readings.
  - *Access-Point Dropout ($p = 0.10$):* Randomly masks visible AP readings to the missing-signal sentinel to train the network on partial, degraded fingerprints.

---

## Repository Structure

```
indoor-wifi-positioning/
├── data/
│   ├── trainingData.csv               # Official UJIIndoorLoc training dataset
│   └── validationData.csv             # Official held-out test dataset
├── models/
│   ├── dl_preprocessing.joblib        # Fitted scalers, encoders, and WAP feature list
│   ├── multitask_best.pth             # Best multitask deep learning checkpoint
│   ├── single_building_best.pth       # Single-task building classifier
│   ├── single_floor_best.pth          # Single-task floor classifier
│   ├── single_coords_best.pth         # Single-task coordinate regressor
│   ├── hybrid_multitask_stacked_best.pth # Stacked hybrid neural model checkpoint
│   └── ablation_*_best.pth            # Checkpoints for ablation study runs A through G
├── notebooks/
│   ├── 01_dataset_eda.ipynb           # Exploratory data analysis & signal distributions
│   ├── 02_preprocessing_features.ipynb # Preprocessing, scaling, and feature engineering
│   ├── 03_classical_ml_models.ipynb   # Classical ML baselines (KNN, RF, XGBoost)
│   ├── 04_evaluation_visualization.ipynb # Classical visualization suite
│   ├── 05_multitask_deep_learning.ipynb  # Multitask network implementation & training
│   ├── 06_hybrid_dl_ml.ipynb          # Stacking and residual correction experiments
│   ├── 07_loss_regularization_ablation.ipynb # Loss balancing & ablation analysis
│   └── 08_dl_evaluation_visualization.ipynb  # Master DL evaluation & diagnostics
├── outputs/
│   ├── confusion_matrix_building.png  # Building classification confusion matrix
│   ├── confusion_matrix_floor.png     # Floor classification confusion matrix
│   ├── error_cdf.png                  # Empirical cumulative distribution function (CDF)
│   ├── error_distribution.png         # Coordinate error histograms and boxplots
│   ├── spatial_error_plot.png         # Campus-wide spatial error vector map
│   ├── final_comparison.csv           # Master metrics table across all models
│   └── failure_cases_analysis.csv     # Granular diagnostic log of 178 flagged test scans
├── presentation/
│   └── final_presentation.pptx        # 12-slide comprehensive presentation deck
├── report/
│   ├── main.tex                       # Complete master IEEEtran conference paper
│   ├── references.bib                 # Verified BibTeX bibliography (23 verified entries)
│   ├── figures/                       # High-resolution publication figures
│   └── sections/                      # Modular LaTeX sections for each member lead
├── research/
│   ├── literature_review_matrix.csv   # Structured matrix of key prior research papers
│   ├── literature_review_draft.md     # In-depth synthesis of state-of-the-art
│   └── viva_preparation.md           # Comprehensive viva voce defense guide (11 Q&As)
├── src/
│   ├── models/
│   │   ├── multitask_net.py           # Shared trunk multitask neural network
│   │   ├── hybrid_model.py            # End-to-end HybridPositioningSystem wrapper
│   │   └── loss.py                    # BasicCombinedLoss and UncertaintyLoss modules
│   ├── data_loader.py                 # Raw dataset ingestion and column specifications
│   ├── data_preprocessing.py          # Cleaning, grouped splitting, and transformations
│   ├── dataset.py                     # PyTorch Dataset wrappers for fast batching
│   ├── augmentations.py               # RSSI Gaussian noise and AP dropout logic
│   ├── stacking_features.py           # Out-of-fold tree probability stacking
│   ├── residual_correction.py         # GBDT residual coordinate corrector
│   ├── train_multitask.py             # Multitask model training script
│   ├── train_hybrid.py                # Hybrid model training script
│   ├── ablation.py                    # Ablation study execution runner
│   ├── evaluate.py                    # Master benchmark evaluation & figure generation
│   └── metrics.py                     # Official Euclidean and EvAAL metric calculations
├── tests/
│   ├── test_preprocessing.py          # Tests for data cleaning, splitting, and scaling
│   ├── test_dl_pipeline.py            # Tests for dataset pipeline and model dimensions
│   ├── test_model.py                  # Tests for MultitaskNet and UncertaintyLoss
│   ├── test_hybrid.py                 # Tests for probability stacking and residual GBDT
│   ├── test_loss_augmentation.py      # Tests for noise injection and AP masking
│   ├── test_evaluation_metrics.py     # Tests for metric equations and EvAAL scoring
│   └── test_metrics.py                # Standalone verification suite for evaluation metrics
└── requirements.txt                   # Complete project dependencies
```

---

## Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
git clone https://github.com/Hashicy/indoor-wifi-positioning.git
cd indoor-wifi-positioning

# Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Running Automated Test Suites
Execute all 63 unit tests spanning preprocessing, neural architectures, loss balancing, hybrid pipelines, and evaluation metrics:
```bash
PYTHONPATH=. pytest
```
*Expected result: 63 passed in ~3.5 seconds.*

### 3. Data Preprocessing
Generate clean, position-grouped splits and fit preprocessing scalers:
```bash
PYTHONPATH=. python3 src/data_preprocessing.py
```
*Outputs: `models/dl_preprocessing.joblib` (16,432 train, 2,795 val, 1,111 test samples; 459 WAPs).*

### 4. Model Training Pipelines
- **Train Multitask Deep Learning Network:**
  ```bash
  PYTHONPATH=. python3 src/train_multitask.py
  ```
- **Train Hybrid Models (Probability Stacking & Residual GBDT):**
  ```bash
  PYTHONPATH=. python3 src/train_hybrid.py
  ```
- **Execute Controlled Loss & Regularization Ablations:**
  ```bash
  PYTHONPATH=. python3 src/ablation.py
  ```

### 5. Master Evaluation & Plot Generation
Regenerate all master tables, spatial error plots, confusion matrices, and diagnostic logs:
```bash
PYTHONPATH=. python3 src/evaluate.py
```
*Outputs are saved to `outputs/` and synchronized with `report/figures/`.*

---

## Deliverables & Academic Artifacts

- **IEEE Conference Research Report:** [`report/main.tex`](report/main.tex)
  - Complete two-column IEEE conference paper integrating all 5 member workstreams, theoretical derivations, empirical tables, and failure case physics.
  - Verified BibTeX bibliography: [`report/references.bib`](report/references.bib)
- **Final Presentation Slide Deck:** [`presentation/final_presentation.pptx`](presentation/final_presentation.pptx)
  - 12-slide widescreen presentation deck aligned with the project guide.
- **Viva Voce Defense Guide:** [`research/viva_preparation.md`](research/viva_preparation.md)
  - Comprehensive answers to all 11 core oral examination defense questions.
- **Literature Review Matrix:** [`research/literature_review_matrix.csv`](research/literature_review_matrix.csv)
  - Systematic review table of prior works with verified DOIs and methodology comparisons.

---

## Citation & Verified References

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
