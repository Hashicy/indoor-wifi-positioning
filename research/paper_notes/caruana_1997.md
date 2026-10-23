# Paper Notes: Multitask Learning (Caruana, 1997)

## Bibliographic Details

| Field | Value |
|-------|-------|
| **Title** | Multitask Learning |
| **Author** | Rich Caruana |
| **Venue** | Machine Learning, Volume 28, Issue 1, pp. 41–75 |
| **Year** | 1997 |
| **DOI** | [10.1023/A:1007379606734](https://doi.org/10.1023/A:1007379606734) |
| **Publisher** | Kluwer Academic Publishers (now Springer) |

---

## Core Thesis

Multitask learning (MTL) is an **inductive transfer** mechanism that improves generalization by training a model on **multiple related tasks in parallel**, using a **shared representation**. The training signals from related tasks act as an inductive bias that helps the model prefer hypotheses that generalize well across all tasks.

---

## Key Contributions (Verified)

### 1. Definition of Multitask Learning
- MTL trains a single model (specifically a neural network in the original work) on multiple tasks simultaneously
- Tasks share hidden layer representations while having **separate output nodes**
- The extra tasks serve as an **inductive bias** — they constrain the hypothesis space to representations useful for multiple problems

### 2. Five Mechanisms Explaining Why MTL Works

| Mechanism | Description |
|-----------|-------------|
| **Data Amplification** | Related tasks effectively increase the sample size for the shared representation, reducing overfitting |
| **Attention Focusing** | Auxiliary tasks help the network focus on features relevant to the main task by providing additional gradient signals |
| **Eavesdropping** | Features that are hard for task A to learn directly may be easy for task B; sharing representations lets A "eavesdrop" on B's learned features |
| **Representation Bias** | MTL prefers representations that are useful across tasks — a form of Occam's razor favoring simpler, more general hypotheses |
| **Regularization** | Training on multiple tasks simultaneously acts as a regularizer, reducing the risk of overfitting to any single task |

### 3. Empirical Demonstrations
- Tested across **multiple domains** including:
  - Backpropagation neural networks
  - K-Nearest Neighbor
  - Kernel regression
- MTL **frequently outperformed** single-task learning (STL) across these domains

---

## Architecture: Hard Parameter Sharing

The paper primarily uses **hard parameter sharing**:

```
Input Layer
    ↓
[Shared Hidden Layers]  ← learned representation used by ALL tasks
    ↓          ↓         ↓
Task A      Task B    Task C    ← separate output heads
```

- All tasks share the **same hidden layer weights**
- Each task has its **own output unit(s)**
- Backpropagation from all tasks flows through the shared layers
- This forces the hidden representation to encode features useful for all tasks

---

## Relevance to Our Project

### Direct Application
Our project has a natural multitask structure:

| Task | Type | Output |
|------|------|--------|
| Building prediction | Classification (3 classes) | Softmax → BuildingID |
| Floor prediction | Classification (5 classes) | Softmax → FloorID |
| Longitude prediction | Regression | Linear → continuous value |
| Latitude prediction | Regression | Linear → continuous value |

### Why MTL Should Help (Applying Caruana's Mechanisms)

1. **Data amplification:** All 4 tasks share the same input fingerprints → the shared layers see 4× the gradient signal
2. **Attention focusing:** Building and floor classification help the coordinate regression focus on location-discriminative WAPs
3. **Eavesdropping:** Features learned for building/floor discrimination (which WAPs are building-specific) are useful for coordinate regression, and vice versa
4. **Representation bias:** The model must learn a WiFi fingerprint representation that simultaneously supports both categorical (building/floor) and continuous (coordinate) predictions
5. **Regularization:** Training 4 tasks prevents the network from overfitting to noise in any single prediction target

### Combined Loss Function (Following Caruana's Framework)
```
L_total = α · L_building(cross-entropy) + β · L_floor(cross-entropy) + γ · L_coordinates(MSE)
```
Where α, β, γ are weights balancing the task losses.

---

## Limitations and Context

- The paper is from **1997** — well before the deep learning revolution
- Experiments used **shallow neural networks** (by modern standards)
- The theory is general, but **auxiliary task selection** remains largely manual and domain-specific
- The paper does not address:
  - How to weight losses across tasks with different scales (addressed later by GradNorm, Chen et al. 2018)
  - Deep architectures (addressed later by Ruder, 2017)
  - Negative transfer (when tasks interfere rather than help)

---

## Connection to Modern MTL References

| Concept from Caruana (1997) | Modern Development |
|---|---|
| Hard parameter sharing | Still dominant in practice (Ruder 2017) |
| Fixed equal loss weighting | Replaced by adaptive weighting: uncertainty-based (Kendall et al. 2018), GradNorm (Chen et al. 2018) |
| Shallow networks | Deep MTL with CNNs, transformers |
| Manual auxiliary task selection | Automated task relationship learning |

---

## Questions for Further Investigation

1. What specific auxiliary task selection heuristics did Caruana recommend?
2. Did Caruana observe negative transfer in any experiments, and how was it handled?
3. What is the exact training procedure — do all tasks start training simultaneously, or are some phased?

---

*Last updated: 2026-10-21*
*Status: Core contributions verified via DOI, Semantic Scholar, and web search. Specific experimental numbers should be confirmed from the full paper text.*
