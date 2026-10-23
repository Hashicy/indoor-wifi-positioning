# Research Gap Analysis

> **Important:** The gap described below is a **hypothesis to validate against the literature**, not a confirmed claim of novelty. It must be verified by checking whether existing work already addresses the specific combination described here.

---

## Context: What the Literature Shows (Verified)

### 1. Classical ML Approaches to WiFi Indoor Localization
- **KNN/WKNN** is the dominant baseline for WiFi fingerprint matching (Torres-Sospedra et al. 2014, 2015)
- **Random Forest, SVM, Decision Trees** have been applied to building/floor classification with PCA preprocessing (Salamah et al. 2016)
- **XGBoost** has been applied to WiFi fingerprint classification in recent work
- These methods typically treat building, floor, and coordinate prediction as **separate, independent tasks**

### 2. Deep Learning Approaches
- **DNNs with stacked autoencoders (SAE)** have been used for hierarchical multi-building/multi-floor localization on UJIIndoorLoc (Kim et al. 2018)
- **CNNs** have been applied to WiFi fingerprinting, treating RSSI vectors as 1D signals (Song et al. 2019)
- **Simple DNNs** have achieved high building/floor classification accuracy (Nowicki & Wietrzykowski 2017) but without coordinate regression

### 3. Multitask Learning Theory
- MTL improves generalization via shared representations when tasks are related (Caruana 1997)
- Hard parameter sharing is the most common MTL architecture (Ruder 2017)
- Loss balancing across tasks with different scales is an open challenge, addressed by methods like GradNorm (Chen et al. 2018)

---

## Hypothesized Research Gap

### The Gap (Hypothesis — Needs Validation)

**Most existing work on WiFi fingerprint-based indoor localization on the UJIIndoorLoc dataset treats the prediction targets in one of these ways:**

1. **Classification only:** Predict building and floor as separate classification problems (ignoring coordinate regression)
2. **Regression only:** Predict longitude and latitude coordinates (ignoring building/floor as auxiliary signals)
3. **Hierarchical/sequential:** Predict building first, then floor, then coordinates in a pipeline (but without joint end-to-end training with a shared representation)
4. **Comparison without integration:** Compare classical ML and deep learning separately, without a systematic comparison that also explores MTL

**What may be underexplored:**
- A **unified multitask deep learning model** that jointly predicts building (classification), floor (classification), and coordinates (regression) through a **single shared representation with explicit multitask loss optimization**
- A **systematic comparison** on the same dataset and preprocessing pipeline that benchmarks:
  - Classical ML baselines (KNN/WKNN, Random Forest, XGBoost) for each task
  - Single-task deep learning
  - Multitask deep learning with combined classification + regression loss
- Explicit evaluation of **whether MTL improves coordinate prediction accuracy by leveraging building/floor classification as auxiliary tasks** (applying Caruana's eavesdropping and attention focusing mechanisms to WiFi localization)

---

## Validation Checklist

Before claiming this gap, the following must be confirmed:

| Check | Status |
|-------|--------|
| Does Kim et al. (2018) use formal MTL with combined loss, or sequential hierarchical prediction? | **[NEEDS VERIFICATION]** — their paper uses hierarchical DNN but the loss structure needs direct confirmation |
| Are there existing papers that explicitly apply hard-parameter-sharing MTL (à la Caruana/Ruder) to UJIIndoorLoc with combined classification + regression loss? | **[NEEDS VERIFICATION]** — no such paper found in our search, but absence of evidence is not evidence of absence |
| Does Song et al. (2019) CNNLoc use multitask outputs or separate models? | **[NEEDS VERIFICATION]** — need to check if their CNN architecture has multiple output heads |
| Are there recent (2022–2026) papers that have already addressed this exact combination? | **[NEEDS VERIFICATION]** — the Feng et al. (2022) survey may contain relevant references |
| Is the XGBoost + MTL comparison already published? | **[NEEDS VERIFICATION]** — no such paper found, but further search needed |

---

## How to Frame the Contribution (Draft)

### If the gap is validated:
> "While prior work has applied classical machine learning and deep learning to WiFi fingerprint-based indoor localization, relatively few studies have systematically compared these approaches and explored whether a unified multitask deep learning model — jointly optimizing building classification, floor classification, and coordinate regression through a shared representation — can outperform both single-task classical and deep learning baselines on the UJIIndoorLoc dataset."

### If similar work exists:
> Reframe as: "We reproduce and extend [existing work] by [specific additional contribution: e.g., adding XGBoost baseline, exploring different loss weighting strategies, evaluating on additional metrics]."

---

## Related Research Questions

1. Does multitask learning (jointly predicting building, floor, and coordinates) improve coordinate prediction accuracy compared to single-task coordinate regression?
2. Which classical ML model (KNN, WKNN, Random Forest, XGBoost) provides the strongest baseline for each prediction target?
3. How should the combined loss function be weighted (fixed weights vs. adaptive methods like GradNorm)?
4. Does the shared representation in MTL learn building/floor-specific RSSI patterns that benefit coordinate regression (i.e., does Caruana's "eavesdropping" mechanism apply)?

---

*Last updated: 2026-10-21*
*Status: Gap is a working hypothesis. 5 verification checks remain before this can be stated as a confirmed contribution.*
