# Mathematical Problem Formulation

## Overview

This document formally defines the indoor WiFi positioning problem as three interrelated prediction tasks: building classification, floor classification, and coordinate regression. It then formulates the multitask learning objective that unifies these tasks.

---

## 1. Input Representation

### WiFi Fingerprint Vector

Let a single WiFi fingerprint observation be a vector:

$$\mathbf{x} = [r_1, r_2, \ldots, r_N] \in \mathbb{R}^N$$

where:
- $N = 520$ (the number of detected Wireless Access Points in UJIIndoorLoc)
- $r_i$ is the Received Signal Strength Indicator (RSSI) value for the $i$-th WAP
- $r_i \in \{-104, -103, \ldots, -1, 0\}$ when the WAP is detected (in dBm)
- $r_i = +100$ when the WAP is not detected (a sentinel value requiring preprocessing)

### Preprocessing (Conceptual)

Before modeling, the raw fingerprints are transformed:

$$\mathbf{x}' = \phi(\mathbf{x})$$

Common preprocessing steps include:
1. Replacing the sentinel value +100 with a floor value (e.g., −110 dBm)
2. Normalizing to [0, 1] via min-max scaling: $x'_i = \frac{r_i - r_{\min}}{r_{\max} - r_{\min}}$
3. Optionally filtering WAPs with few detections (feature selection)

---

## 2. Prediction Targets

The dataset associates each fingerprint $\mathbf{x}$ with four prediction targets:

### Task 1: Building Classification

$$y_b \in \{0, 1, 2\}$$

Three buildings at Universitat Jaume I. This is a **3-class classification** problem.

### Task 2: Floor Classification

$$y_f \in \{0, 1, 2, 3, 4\}$$

Floor level within the building. This is a **5-class classification** problem.

### Task 3: Coordinate Regression

$$\mathbf{y}_c = (\text{longitude}, \text{latitude}) \in \mathbb{R}^2$$

Continuous spatial coordinates in the local coordinate system of the university campus. This is a **2-output regression** problem.

---

## 3. Single-Task Formulations

### 3a. Classification (Building or Floor)

For a classification task with $C$ classes, we learn a function:

$$f: \mathbb{R}^N \rightarrow \Delta^{C-1}$$

that maps a fingerprint to a probability distribution over classes, where $\Delta^{C-1}$ is the $(C-1)$-simplex.

**Loss function:** Categorical Cross-Entropy

$$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{M} \sum_{j=1}^{M} \sum_{c=1}^{C} y_{j,c} \cdot \log(\hat{y}_{j,c})$$

where:
- $M$ is the number of training samples
- $y_{j,c}$ is the one-hot encoded true label (1 if sample $j$ belongs to class $c$, else 0)
- $\hat{y}_{j,c} = \text{softmax}(f(\mathbf{x}_j))_c$ is the predicted probability for class $c$

**Evaluation metrics:**
- **Accuracy:** Fraction of correctly predicted classes
- **Building hit rate:** $\frac{\text{Correct building predictions}}{M_{\text{test}}}$
- **Floor hit rate:** $\frac{\text{Correct floor predictions}}{M_{\text{test}}}$

### 3b. Regression (Coordinates)

For coordinate regression, we learn:

$$g: \mathbb{R}^N \rightarrow \mathbb{R}^2$$

that maps a fingerprint to a (longitude, latitude) pair.

**Loss function:** Mean Squared Error (MSE)

$$\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{M} \sum_{j=1}^{M} \|\mathbf{y}_{c,j} - \hat{\mathbf{y}}_{c,j}\|^2$$

$$= \frac{1}{M} \sum_{j=1}^{M} \left[ (\text{lon}_j - \hat{\text{lon}}_j)^2 + (\text{lat}_j - \hat{\text{lat}}_j)^2 \right]$$

**Evaluation metrics:**
- **Mean positioning error (MPE):**

$$\text{MPE} = \frac{1}{M_{\text{test}}} \sum_{j=1}^{M_{\text{test}}} \sqrt{(\text{lon}_j - \hat{\text{lon}}_j)^2 + (\text{lat}_j - \hat{\text{lat}}_j)^2}$$

Note: Since the coordinates are in a local system (not geographic degrees), Euclidean distance is appropriate (no Haversine formula needed).

---

## 4. Multitask Learning Formulation

### 4a. Shared Representation (Hard Parameter Sharing)

Following Caruana (1997), we define a multitask neural network with:

1. **Shared encoder** $h_{\theta_s}$: maps input fingerprints to a shared latent representation
   
   $$\mathbf{z} = h_{\theta_s}(\mathbf{x}') \in \mathbb{R}^d$$

2. **Task-specific heads:**
   - Building head: $f_b(\mathbf{z}; \theta_b) \rightarrow \Delta^2$ (3-class softmax)
   - Floor head: $f_f(\mathbf{z}; \theta_f) \rightarrow \Delta^4$ (5-class softmax)
   - Coordinate head: $g_c(\mathbf{z}; \theta_c) \rightarrow \mathbb{R}^2$ (linear output)

### Architecture Diagram

```
WiFi Fingerprint x ∈ R^520
         ↓
   [Preprocessing φ]
         ↓
    x' ∈ R^N'
         ↓
  ┌─────────────────┐
  │  Shared Encoder  │  ← θ_s (shared parameters)
  │  h(x') → z      │
  └───┬─────┬────┬──┘
      ↓     ↓    ↓
  ┌──────┐ ┌────┐ ┌──────────┐
  │Build │ │Flr │ │  Coord   │
  │Head  │ │Head│ │  Head    │
  │θ_b   │ │θ_f │ │  θ_c     │
  └──┬───┘ └─┬──┘ └────┬─────┘
     ↓       ↓         ↓
  ŷ_b ∈ Δ²  ŷ_f ∈ Δ⁴  ŷ_c ∈ R²
  (softmax)  (softmax)  (linear)
```

### 4b. Combined Loss Function

The total multitask loss is a weighted sum of the individual task losses:

$$\mathcal{L}_{\text{total}}(\theta_s, \theta_b, \theta_f, \theta_c) = \alpha \cdot \mathcal{L}_{\text{building}} + \beta \cdot \mathcal{L}_{\text{floor}} + \gamma \cdot \mathcal{L}_{\text{coords}}$$

where:

$$\mathcal{L}_{\text{building}} = -\frac{1}{M} \sum_{j=1}^{M} \sum_{c=0}^{2} y_{j,c}^{(b)} \log \hat{y}_{j,c}^{(b)}$$

$$\mathcal{L}_{\text{floor}} = -\frac{1}{M} \sum_{j=1}^{M} \sum_{c=0}^{4} y_{j,c}^{(f)} \log \hat{y}_{j,c}^{(f)}$$

$$\mathcal{L}_{\text{coords}} = \frac{1}{M} \sum_{j=1}^{M} \|\mathbf{y}_{c,j} - \hat{\mathbf{y}}_{c,j}\|^2$$

### 4c. Loss Weighting Strategies

The hyperparameters $\alpha, \beta, \gamma$ control the relative importance of each task:

| Strategy | Description |
|----------|-------------|
| **Equal weighting** | $\alpha = \beta = \gamma = 1$ (simple but ignores scale differences) |
| **Scale-normalized** | Set weights inversely proportional to the magnitude of each loss to balance gradient contributions |
| **Grid/random search** | Tune $\alpha, \beta, \gamma$ as hyperparameters on validation set |
| **GradNorm** (Chen et al. 2018) | Dynamically adjust weights during training based on gradient norms to equalize learning rates across tasks |

**Note:** Classification losses (cross-entropy) and regression losses (MSE) operate on different scales. Without proper weighting, one task may dominate training. This is a known challenge in MTL.

---

## 5. Optimization

All parameters $\Theta = \{\theta_s, \theta_b, \theta_f, \theta_c\}$ are trained jointly via backpropagation:

$$\Theta^* = \arg\min_\Theta \mathcal{L}_{\text{total}}(\Theta)$$

Using stochastic gradient descent or an adaptive optimizer (e.g., Adam):

$$\Theta_{t+1} = \Theta_t - \eta \nabla_\Theta \mathcal{L}_{\text{total}}(\Theta_t)$$

Key: Gradients from **all three tasks** flow through the shared encoder $h_{\theta_s}$, which is the mechanism by which MTL provides its benefits (per Caruana 1997).

---

## 6. Evaluation Framework

### Metrics per Task

| Task | Metric | Formula |
|------|--------|---------|
| Building | Accuracy (hit rate) | $\frac{1}{M}\sum_{j} \mathbb{1}[\hat{y}_b = y_b]$ |
| Floor | Accuracy (hit rate) | $\frac{1}{M}\sum_{j} \mathbb{1}[\hat{y}_f = y_f]$ |
| Coordinates | Mean Positioning Error | $\frac{1}{M}\sum_{j}\sqrt{(\Delta\text{lon})^2 + (\Delta\text{lat})^2}$ |

### Comparison Structure

The project compares:
1. **Classical ML baselines** (KNN, WKNN, Random Forest, XGBoost) — trained independently per task
2. **Single-task DNN** — separate deep networks per task
3. **Multitask DNN** — joint model with shared representation and combined loss

The key question: **Does the multitask model outperform single-task models, particularly for coordinate regression, by leveraging building/floor classification as auxiliary tasks?**

---

## 7. Research Questions (Formal)

1. **RQ1:** How do classical ML models (KNN, WKNN, Random Forest, XGBoost) compare against deep learning models for building classification, floor classification, and coordinate regression on the UJIIndoorLoc dataset?

2. **RQ2:** Does a multitask deep learning model with a shared representation and combined classification + regression loss achieve lower mean positioning error than an equivalent single-task deep regression model?

3. **RQ3:** How does the choice of loss weighting strategy ($\alpha, \beta, \gamma$) affect the performance of the multitask model across all prediction targets?

4. **RQ4:** What RSSI preprocessing and feature selection strategies yield the best performance for both classical ML and deep learning approaches?

---

*Last updated: 2026-10-21*
*Note: All formulations are standard mathematical definitions. The specific architectural choices (number of layers, hidden units, activation functions) will be determined experimentally.*
