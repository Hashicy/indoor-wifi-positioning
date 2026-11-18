# Comprehensive Viva Voce Preparation Guide
## Indoor Positioning via Wi-Fi Signal Strength Features and Deep Multitask Learning
**Project Team:** Five-Member Deep Learning Implementation (Team T076)  
**Lead Responsibility:** Member 5 — Literature Review, Theoretical Foundations, System Integration & Final Defense  

---

## Executive Summary of Core Benchmark Results
Before answering individual theoretical questions, memorize these baseline facts from our strictly held-out test set ($N = 1{,}111$, 9 unseen smartphone models, 4-month temporal gap):

| Metric | Classical Baseline (Random Forest) | Multitask Neural Net (Phase 2 DL) | Full Hybrid System (Stacked DL + GBDT) |
| :--- | :---: | :---: | :---: |
| **Building Accuracy** | 99.73% | 99.73% | **99.82%** (only 2 errors / 1,111) |
| **Floor Accuracy** | 89.92% | 91.00% | **92.08%** |
| **Joint Bldg+Floor Acc** | 89.74% | 91.00% | **92.08%** (100% building when floor is right) |
| **Mean Coordinate Error** | 12.30 m | 8.95 m (-27.2%) | **8.51 m** (-30.8% error reduction) |
| **Median Coordinate Error**| 8.15 m | 7.04 m | **6.30 m** (-22.7% error reduction) |
| **90th Percentile ($P_{90}$)** | 29.55 m | 17.87 m | **17.58 m** (-40.5% tail error bound) |
| **EvAAL Competition Score**| 13.04 | 9.50 | **8.96** (Best overall score) |
| **Inference Latency (CPU)**| ~12 ms | **<0.8 ms** (15x faster) | ~2.5 ms |

---

## Detailed Viva Questions & Model Defense Answers

### 1. Why choose multitask learning instead of three separate models?
**Key Points:**
- **Inductive Bias & Feature Sharing (Caruana, 1997):** By learning to predict building and floor alongside coordinates, the shared hidden layers are forced to learn macro-scale spatial representations (campus geometry and concrete slab separations). These act as an inductive regularizer for the continuous coordinate regression head.
- **Elimination of Cascading Error Compounding:** In traditional hierarchical positioning (e.g., Kim et al., 2018), a pipeline predicts Building $\to$ Floor $\to$ Coordinates. If the initial building or floor classifier makes a mistake, downstream coordinate prediction fails catastrophically (hundreds of meters off). In our hard-parameter-sharing network, all three heads branch in parallel from a shared trunk, preventing single-point failure propagation.
- **Computational & Edge Efficiency:** A single forward pass through our 512-256-128 network (~400K parameters) computes all three outputs simultaneously in under 0.8 ms on a mobile CPU. In contrast, three separate deep networks require 3x memory footprint and 3x forward-pass FLOPs.
- **Empirical Reality Check:** In our controlled experiments across 3 random seeds (seeds 42, 7, 123), multitask learning achieved virtually identical floor accuracy (91.21% vs 91.51%) and coordinate error (9.03 m vs 8.59 m) while operating within a unified representation. Loss scale disparities (cross-entropy dominating MSE) explain why unweighted multitask does not wildly exceed single-task, motivating our uncertainty weighting studies.

---

### 2. Why use RSSI values for indoor positioning?
**Key Points:**
- **Zero Additional Hardware Infrastructure:** Wi-Fi access points are already deployed ubiquitously across commercial buildings, hospitals, university campuses, and transit hubs. Fingerprinting uses ambient IEEE 802.11 beacon transmissions without requiring costly hardware anchor nodes (unlike Ultra-Wideband / UWB) or specialized sensors.
- **Bypasses Explicit Propagation Modeling:** Indoor environments are riddled with complex multipath reflections, non-line-of-sight propagation, and dynamic scattering from human bodies and furniture. Parametric path-loss equations (e.g., Log-Distance Path Loss) fail because they require exact 3D coordinates, antenna radiation patterns, and wall transmission coefficients. Fingerprinting treats localization as an empirical pattern recognition problem, mapping the observed radio signature directly to spatial targets.
- **Omnipresent Standard on Mobile Devices:** Every smartphone operating system (Android, iOS) exposes Wi-Fi scan results containing BSSID and decibel RSSI values, making this approach immediately deployable to billions of mobile users.

---

### 3. Why is -110 used for unavailable access points?
**Key Points:**
- **The Raw Sentinel Problem (+100):** In UJIIndoorLoc, unobserved APs are coded as `+100`. In reality, genuine RSSI values strictly range from $-104$ dBm (receiver sensitivity threshold) up to $0$ dBm (antenna touching the AP).
- **Catastrophic Distortions of Raw +100:**
  1. *In Neural Networks:* A value of $+100$ acts as an enormous positive stimulus, falsely indicating an insanely powerful transmitter right next to the phone.
  2. *In KNN:* The distance between unobserved ($+100$) and strong ($-40$) is $|100 - (-40)| = 140$, completely warping Euclidean signal distances.
  3. *In Tree Models:* Breaks physical monotonicity: strong ($-30$) $>$ moderate ($-70$) $>$ weak ($-100$) $>$ undetected ($+100$ is non-monotonic).
- **Physical Justification for -110 dBm:**
  Mapping unobserved APs to $-110$ dBm places absent signals strictly below the physical hardware sensitivity noise floor ($-104$ dBm).
  When scaled with `MinMaxScaler` fitted on training data:
  $$\tilde{x} = \frac{r - (-110)}{0 - (-110)}$$
  Absent access points cleanly map to $0.0$, weak detected signals map to small positive values ($0.05\text{--}0.20$), and the strongest signals approach $1.0$. This maintains monotonic physical radio ordering.

---

### 4. Why use GELU and LayerNorm?
**Key Points:**
- **Why GELU (Gaussian Error Linear Unit) over ReLU:**
  - ReLU ($\max(0, x)$) has a hard zero derivative for all $x < 0$. With $>96.5\%$ feature sparsity, large numbers of neurons receive zero inputs, causing the "dying ReLU" phenomenon where neurons permanently stop updating.
  - GELU scales activations smoothly according to the standard Gaussian cumulative distribution: $\text{GELU}(z) = z \cdot \Phi(z) = z \cdot P(Z \le z)$. For negative inputs, GELU maintains small, smooth non-zero gradients, preventing dead units and creating smooth continuous manifolds ideal for continuous coordinate regression.
- **Why LayerNorm over BatchNorm:**
  - BatchNorm standardizes across the mini-batch dimension. In sparse Wi-Fi scans, mini-batch statistics fluctuate erratically depending on which specific buildings or floors are sampled in each batch.
  - LayerNorm computes the mean and variance across the feature dimension independently for each sample:
    $$\hat{z} = \frac{z - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} \cdot \gamma + \beta$$
  - This provides consistent internal covariate stabilization regardless of batch size or batch class composition, stabilizing deep multi-layer backpropagation.

---

### 5. Why combine Random Forest or boosting with a neural network?
**Key Points:**
- **Complementary Inductive Biases:**
  - *Tree Models (Random Forest / GBDT):* Excel at orthogonal axis-aligned partition cuts ($\mathbb{I}(x_j \le \theta)$). They naturally isolate discrete structural boundaries (buildings, reinforced concrete floor slabs) where radio attenuation is sharp and non-linear. However, in continuous space, trees predict piecewise-constant step functions, causing quantization errors inside rooms.
  - *Deep Neural Networks:* Excel at learning smooth, continuous metric coordinate manifolds, but can struggle with high-dimensional sparsity ($459$ sparse inputs) and discrete floor transitions.
- **Hybrid Synergy:**
  - **Probability Stacking:** Feeds tree-derived class probabilities ($p_{\text{bld}}, p_{\text{flr}}$) into the neural trunk, providing explicit discrete structural priors.
  - **Residual Boosting:** Captures systematic local spatial distortions that the smooth global neural manifold smooths over.
- **Empirical Validation:** The Full Hybrid system beats both pure Random Forest (mean error 8.51 m vs 12.30 m) and pure Multitask DL (8.51 m vs 8.95 m), achieving the lowest EvAAL score of 8.96.

---

### 6. What is residual correction?
**Key Points:**
- **Definition:** For any training scan with true coordinates $\mathbf{y} = (x, y)$ and deep learning neural prediction $\hat{\mathbf{y}}_{\text{DL}} = (\hat{x}_{\text{DL}}, \hat{y}_{\text{DL}})$, the residual spatial error vector is defined as:
  $$e_x = x - \hat{x}_{\text{DL}}, \quad e_y = y - \hat{y}_{\text{DL}}$$
- **Mechanism:** A gradient boosted decision tree (GBDT regressor) is trained to predict $(\hat{e}_x, \hat{e}_y)$ directly from the input RSSI features.
- **Inference Correction:** At test time, the final coordinate prediction is:
  $$\hat{x}_{\text{final}} = \hat{x}_{\text{DL}} + \hat{e}_x, \quad \hat{y}_{\text{final}} = \hat{y}_{\text{DL}} + \hat{e}_y$$
- **Why it Works:** Deep neural networks fit a global smooth manifold across the campus. Local physical anomalies—such as an elevator shaft, metallic fire exit door, or reinforced pillar—introduce localized, non-smooth signal distortions. Boosting trees fit these localized spatial residuals without distorting the global neural trunk.

---

### 7. What is data leakage and how was it prevented?
**Key Points:**
- **Definition:** Data leakage occurs when information from the test/validation set (or unallowable metadata) influences model training, producing artificially high training/validation scores that fail to generalize.
- **Three Strict Leakage Prevention Measures in Our Pipeline:**
  1. *Metadata Stripping:* Attributes such as `USERID`, `PHONEID`, `SPACEID`, `RELATIVEPOSITION`, and `TIMESTAMP` were strictly excluded from model feature inputs. Certain users only walked specific corridors; models trained on `USERID` memorize user-building shortcuts rather than learning RF signal mappings.
  2. *Position-Grouped Train/Val Split (`GroupShuffleSplit`):* Each physical reference point was scanned ~20 times by different surveyors. A standard random split places identical fingerprints in both train and validation sets. We grouped by unique spatial coordinates (position ID), ensuring train (16,432) and validation (2,795) share zero positions.
  3. *Out-of-Fold (OOF) Stacking:* When generating tree probabilities for the training set, we used 5-fold stratified cross-validation. The downstream neural net never saw in-sample overfitted tree probabilities.
  4. *Fit-on-Train-Only Scalers:* All `MinMaxScaler`, `StandardScaler`, and `LabelEncoder` objects were fitted exclusively on the training split and only transformed the validation and test sets.

---

### 8. Why can uncertainty weighting be useful?
**Key Points:**
- **The Multitask Loss Scale Mismatch:**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{bld}} + \mathcal{L}_{\text{flr}} + \mathcal{L}_{\text{coords}}$$
  Categorical cross-entropy losses are typically $\approx 0.1\text{--}1.5$, while standardized coordinate MSE is $\approx 0.02\text{--}0.10$. Because the floor loss is 10–20x larger, backpropagation gradients are dominated by floor classification, effectively under-training coordinate regression.
- **Homoscedastic Uncertainty Formulation (Kendall, Gal & Cipolla, CVPR 2018):**
  $$\mathcal{L}_{\text{total}} = \sum_{i} \left[ \exp(-s_i) \mathcal{L}_i + c_i s_i \right]$$
  where $s_i = \log \sigma_i^2$ is a trainable log-variance parameter.
- **Likelihood-Derived Coefficients:**
  - Classification (softmax likelihood): $c_i = 1.0$
  - Regression (Gaussian likelihood): $c_i = 0.5$
- **Adaptive Precision Weighting:** $\exp(-s_i)$ acts as an adaptive precision weight. If task $i$'s loss is excessively noisy or large, $s_i$ increases, down-weighting the task's gradient contribution. The regularization term $c_i s_i$ acts as a penalty that prevents $s_i \to \infty$.

---

### 9. How is Euclidean localization error calculated?
**Key Points:**
- **Mathematical Formula:** For sample $i$ with ground-truth coordinates $(x_i, y_i)$ and predicted coordinates $(\hat{x}_i, \hat{y}_i)$:
  $$d_i = \sqrt{(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}$$
- **Verification of Units:** In UJIIndoorLoc, `LONGITUDE` and `LATITUDE` are projected Universal Transverse Mercator (UTM) metric coordinates. The unit difference corresponds exactly to physical meters.
- **Summary Metrics Computed:**
  - *Mean Error:* $\bar{d} = \frac{1}{N} \sum_{i=1}^N d_i$ (8.51 m for Full Hybrid)
  - *Median Error ($P_{50}$):* 50th percentile (6.30 m) — robust against spatial outliers
  - *75th Percentile ($P_{75}$):* 10.72 m
  - *90th Percentile ($P_{90}$):* 17.58 m — reliability bound for mission-critical LBS
  - *Maximum Error:* Worst-case outlier across the entire campus (78.90 m)
- **Official EvAAL Competition Score:**
  $$\mathcal{S}_{\text{EvAAL}} = \bar{d} + 50 \cdot (1 - \text{Acc}_{\text{bld}}) + 4 \cdot \frac{1}{N}\sum |f_i - \hat{f}_i|$$
  Penalizes wrong building with 50 meters and wrong floor with 4 meters per floor level.

---

### 10. How would the model behave with unseen access points or devices?
**Key Points:**
- **Unseen Access Points:**
  - If a new AP is deployed that was not in the 459 cataloged features, it cannot be processed without updating the input dimension.
  - If an existing AP is turned off or removed, the device observes an absent reading. In our model, this maps to $-110$ dBm ($0.0$ after scaling). Because we specifically applied **Access-Point Dropout ($p=0.10$)** during training, our network was trained to reconstruct spatial coordinates from partial fingerprints where APs are randomly dropped out.
- **Unseen Devices (Device Heterogeneity):**
  - Different smartphone vendors have different antenna gains and casing attenuations, shifting RSSI by up to $15$ dBm.
  - Our test set explicitly contained **9 smartphone models never seen during training**.
  - In our ablation study, **RSSI Gaussian noise augmentation ($\sigma = 0.02$, $\pm 1$--$2$ dBm)** forced the network to learn invariant spatial contours rather than memorizing brittle exact decibel thresholds.
  - While building accuracy remained rock-solid ($99.82\%$), severe hardware gain shifts on Phone ID 20 caused occasional adjacent-floor ambiguities, demonstrating that hardware calibration remains important for industrial deployment.

---

### 11. How do you know that the model generalizes?
**Key Points:**
- **Temporal Holdout Evaluation:**
  Our test set is not an in-sample random split. It was recorded **4 months later** (September–October 2013 vs May–June 2013) by different operators. Over 4 months, seasonal environmental changes, human traffic, and furniture movements alter multipath fading. Achieving $99.82\%$ building accuracy and $8.51$ m mean error on this delayed test set proves robust temporal generalization.
- **Position-Grouped Validation:**
  During training, model selection and early stopping were governed by validation scans that shared zero reference positions with training scans (`GroupShuffleSplit`).
- **Absence of Overfitting Gaps:**
  In our ablation study:
  - Without dropout ($p=0.0$), the model overfit (mean error degraded to $9.49$ m).
  - With dropout ($p=0.3$), test error dropped to $8.85$ m and floor accuracy rose to $91.81\%$, confirming that regularization actively closed the generalization gap.
- **Cross-Seed Reproducibility:**
  Across three distinct random seeds (42, 7, 123), multitask test error was $9.03 \pm 0.10$ m, building accuracy was $99.88 \pm 0.14\%$, and floor accuracy was $91.21 \pm 0.23\%$, showing that generalization is stable and not an artifact of random weight initialization.
