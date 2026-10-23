# Comprehensive Viva Voce Questions and Answers

## Project Title
**Indoor Wi-Fi Positioning Using RSSI Features, Classical Machine Learning, and Deep Multitask Learning**

## Member 1 Focus Areas
- Scientific Motivation & Context
- Theoretical Foundations & Algorithm Selection (KNN, Random Forest, XGBoost, Multitask Neural Networks)
- Evaluation Metrics (Discrete Accuracy vs Continuous Euclidean Positioning Error)
- Dataset Properties & Preprocessing Mechanics (Sparsity, High Dimensionality, Sentinel Values, Device Heterogeneity)
- Research Questions, Objectives, and Scope Limitations

---

## Part 1: Motivation, Context, and Problem Formulation

### Q1: Why is indoor positioning a difficult problem compared to outdoor positioning?
**Answer:**
Outdoors, Global Navigation Satellite Systems (GNSS / GPS) provide reliable 3D localization via line-of-sight trilateration. However, microwave satellite signals (~1.5 GHz) suffer severe attenuation and blockage when penetrating building materials (reinforced concrete, tinted glass, steel frames). Indoors, multipath propagation—where radio frequency (RF) signals reflect, refract, and scatter off walls, furniture, and moving people—destroys time-of-flight geometric assumptions. Consequently, indoor positioning relies on opportunistic terrestrial signals such as Wi-Fi Received Signal Strength Indication (RSSI) or Bluetooth Low Energy (BLE), treating positioning as an empirical pattern-recognition problem rather than a purely geometric one.

### Q2: Why choose Wi-Fi RSSI fingerprinting over propagation modeling or dedicated hardware (UWB)?
**Answer:**
1. **Zero Additional Hardware Infrastructure:** Wi-Fi access points (APs) are ubiquitously deployed in universities, hospitals, airports, and commercial facilities. Localization uses existing IEEE 802.11 beacons without extra capital investment.
2. **Fingerprinting Bypasses Propagation Physics:** Parametric path-loss models (like the Log-Distance Path Loss model) require knowing the exact physical coordinates and transmit power of all APs, and they struggle with complex, non-homogeneous indoor wall layouts. Fingerprinting bypasses explicit channel modeling by mapping observed RSSI vectors directly to empirical spatial coordinates.
3. **Trade-off with UWB:** Ultra-Wideband (UWB) provides centimeter-level accuracy via nanosecond time-difference-of-arrival (TDoA), but requires dedicated anchor nodes and UWB-enabled receiver tags, which is costly and incompatible with legacy mobile devices.

### Q3: What is the formal mathematical formulation of the prediction targets in our project?
**Answer:**
Given a preprocessed RSSI vector $\mathbf{x} \in \mathbb{R}^{520}$, the positioning problem is framed as a simultaneous multi-target prediction:
1. **Building Identification ($y_b$):** Multiclass classification where $y_b \in \{0, 1, 2\}$ ($B = 3$ campus buildings).
2. **Floor Identification ($y_f$):** Multiclass classification where $y_f \in \{0, 1, 2, 3, 4\}$ ($F = 5$ possible floors, with 13 valid building-floor combinations across the site).
3. **2D Coordinates ($\mathbf{y}_c$):** Continuous bivariate regression predicting $\mathbf{y}_c = (x_{\text{long}}, y_{\text{lat}}) \in \mathbb{R}^2$ in projected Universal Transverse Mercator (UTM) metric coordinates.

### Q4: Why is indoor localization formulated as a joint classification and regression task rather than just regression?
**Answer:**
Spatial hierarchy has distinct physical implications. An error in continuous coordinate regression of 15 meters within the same room or corridor is a minor nuisance. However, mispredicting a floor or building—even if Euclidean distance appears moderate in 3D projection—places a user on the wrong level or in a separate structure separated by walls and flights of stairs. By explicitly decoupling discrete building and floor classification from metric coordinate regression, the model preserves topological boundaries while refining spatial resolution.

---

## Part 2: Choice of Algorithms and Modeling Rationale

### Q5: Why is K-Nearest Neighbors (KNN/WKNN) considered the primary baseline in Wi-Fi fingerprinting?
**Answer:**
KNN is the canonical instance-based benchmark established in RADAR (Bahl & Padmanabhan, 2000) and analyzed extensively on UJIIndoorLoc by Torres-Sospedra et al. (2015).
- **Rationale:** If physical space is smoothly covered by reference points, similar spatial locations will observe similar signal patterns from the same set of APs.
- **Weighted KNN (WKNN):** Weights neighbors inversely proportional to distance in signal space ($w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}$), pulling the regression estimate toward the closest signal match.
- **Limitations:** KNN has $\mathcal{O}(M \cdot N)$ inference complexity per query (where $M$ is the radio map size and $N$ is feature dimension), requires storing the entire dataset in memory, and cannot extract latent compositional representations.

### Q6: What advantages do Random Forests offer for RSSI data over linear models and KNN?
**Answer:**
1. **Non-linear Partitioning:** Decision trees perform axis-aligned splits that naturally isolate specific signal strength ranges for informative APs.
2. **Robustness to Missing-Signal Sentinels:** Trees naturally branch on thresholds, effectively isolating the artificial missing-signal indicator (e.g., $r_i \le -105$ dBm) from active measurements without requiring continuous linearity.
3. **Variance Reduction:** Bagging and random feature subsampling reduce model variance caused by noisy RSSI fluctuations and device temporal fading.
4. **Feature Importance:** RF computes Mean Decrease in Impurity (MDI) or permutation importance, identifying which APs contribute most strongly to building or floor boundaries.

### Q7: Why include Gradient Boosted Trees (XGBoost)?
**Answer:**
While Random Forest trains independent trees in parallel to reduce variance, Gradient Boosting (XGBoost) trains shallow trees sequentially to minimize a regularized objective loss:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$$
where $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum w_j^2$.
- **Strengths:** XGBoost excels on tabular, sparse feature sets with complex feature interactions. Second-order Taylor expansions (gradients and Hessians) provide rapid convergence, and built-in sparsity-aware split finding efficiently handles absent APs.
- **Benchmarking Role:** It represents the state-of-the-art tree ensemble method, serving as the strongest classical baseline against deep learning architectures.

### Q8: What is Multitask Learning (MTL) and what is the theoretical justification (Caruana, 1997) for applying it here?
**Answer:**
According to Caruana (1997), Multitask Learning improves generalization by learning tasks in parallel using a shared representation; what is learned for each task can help other tasks be learned better.
In our context, the shared encoder learns a unified latent embedding $\mathbf{z} = g(\mathbf{x}; \mathbf{\Theta}_{\text{shared}})$ from raw 520-dimensional RSSI vectors, branching into task-specific heads:
- Head 1: Building Logits $\hat{\mathbf{y}}_b = h_b(\mathbf{z})$
- Head 2: Floor Logits $\hat{\mathbf{y}}_f = h_f(\mathbf{z})$
- Head 3: Coordinates $\hat{\mathbf{y}}_c = h_c(\mathbf{z})$

**Caruana's Inductive Transfer Mechanisms in our System:**
1. **Data Amplification:** The effective sample size for feature representation increases because gradients from all three tasks simultaneously tune the shared parameters.
2. **Eavesdropping:** The coordinate regression task can "eavesdrop" on features learned to separate buildings and floors, ensuring continuous predictions do not drift across architectural boundaries.
3. **Attention Focusing:** Noisy RSSI fluctuations on irrelevant APs are filtered out because only APs providing mutual utility across building, floor, and coordinate tasks receive consistent positive reinforcement.
4. **Representation Bias:** MTL restricts search space to representations that explain multiple phenomena, decreasing overfitting compared to training an isolated regression network.

### Q9: What is the difference between Hard Parameter Sharing and Soft Parameter Sharing in MTL?
**Answer:**
- **Hard Parameter Sharing (our architecture):** The initial hidden layers (dense layers with batch normalization and dropout) are shared identically across all tasks, and only the final output layers are task-specific. It provides strong regularization and drastically reduces memory and inference footprint ($\mathcal{O}(|\mathbf{\Theta}_{\text{shared}}|)$).
- **Soft Parameter Sharing:** Each task has its own separate neural network with its own parameters, and the distance between corresponding layer weights is constrained via regularization (e.g., $\ell_2$ distance or trace norm). It is computationally heavier and less practical for embedded mobile localization.

---

## Part 3: Evaluation Metrics and Scientific Rigor

### Q10: What is the fundamental difference between Classification Accuracy and Mean Positioning Error (MPE)?
**Answer:**
- **Classification Accuracy (Discrete):**
  $$\text{Acc} = \frac{1}{M}\sum_{i=1}^M \mathbb{I}(\hat{y}_i = y_i)$$
  Measures the exact hit rate for categorical targets (Building, Floor). It is zero-one: misidentifying Floor 2 as Floor 3 has the exact same penalty as misidentifying Floor 2 as Floor 0.
- **Mean Positioning Error (MPE / 2D Euclidean Distance):**
  $$\text{MPE} = \frac{1}{M}\sum_{i=1}^M \sqrt{(\hat{x}_i - x_i)^2 + (\hat{y}_i - y_i)^2}$$
  A continuous metric measured in physical metres. It penalizes errors proportionally to spatial displacement.
- **Why Both are Critical:** A model could achieve 99% building and floor accuracy but exhibit poor coordinate regression (e.g., 18m MPE within the floor). Conversely, a pure regression model might achieve 6m average error on a 2D Euclidean plane but frequently predict points outside the building or on the wrong floor.

### Q11: What is the official IPIN Competition positioning error metric, and how does it penalize floor/building errors?
**Answer:**
In the EvAAL / IPIN competition framework on UJIIndoorLoc, error is calculated as a 3D penalised metric:
$$\text{Error}_{\text{IPIN}} = \sqrt{(\hat{x} - x)^2 + (\hat{y} - y)^2} + p_f \cdot |\hat{y}_f - y_f| + p_b \cdot \mathbb{I}(\hat{y}_b \neq y_b)$$
where penalties (typically $p_f = 4\text{ m}$ or $5\text{ m}$ per floor and $p_b = 50\text{ m}$ for incorrect building) reflect the severe user inconvenience of traveling between vertical levels or separate buildings.

### Q12: Why should cumulative distribution functions (CDF) of positioning error be reported in addition to mean error?
**Answer:**
Mean Positioning Error (MPE) is sensitive to extreme outliers (e.g., a single prediction with 120m error significantly inflates the mean). The Error CDF plots positioning error $e$ on the x-axis against $P(E \le e)$ on the y-axis, providing key operational percentiles:
- **Median Error (50th percentile):** Represents typical expected user accuracy.
- **90th / 95th Percentile Error:** Indicates system dependability under worst-case signal degradation or AP shadowing.

---

## Part 4: Dataset Properties, Preprocessing, and Pitfalls

### Q13: Describe the UJIIndoorLoc dataset dimensions, sparsity, and collection conditions.
**Answer:**
- **Dimensions:** 19,937 training rows and 1,111 validation rows; 520 WAP columns plus 9 context/target attributes (529 total columns).
- **Physical Scale:** Universitat Jaume I campus, 3 buildings (Buildings 0, 1, 2) totaling ~110,000 m², up to 5 levels (Floors 0–4).
- **Sparsity:** ~96.5% of RSSI entries are the unobserved sentinel value $+100$. A typical scan detects only 15–17 access points out of 520.
- **Heterogeneity:** Collected by over 20 distinct users utilizing 25 different Android mobile phone models over a span of several months.

### Q14: Why is $+100$ used as the unobserved sentinel in UJIIndoorLoc, and what happens if you feed it raw to an algorithm?
**Answer:**
Genuine RSSI values range from $-104$ dBm (minimum sensitivity limit) up to $0$ dBm (extremely close proximity). The original authors encoded unobserved APs as $+100$ so that missingness is easily distinguishable from real measurements.
**Consequences of unhandled $+100$:**
1. **In Linear Models / Neural Networks:** $+100$ acts as a massive positive stimulus, causing the network to treat an undetected AP as a transmitter emitting immense radio energy, distorting weight updates.
2. **In KNN:** The Euclidean distance between an unobserved AP ($+100$) and a strong signal ($-40$) is $|100 - (-40)| = 140$, whereas between two unobserved APs it is $0$. If left unscaled, distant unobserved APs dominate the geometric distance calculation.
3. **In Tree Models:** Although trees can split on $x > 0$, raw $+100$ inverts physical ordering ($-50 > -90 > +100$ is non-monotonic).

### Q15: How should the sentinel value be preprocessed?
**Answer:**
1. **Replacement with Floor Value:** Map $+100 \to -110$ dBm (or $-105$ dBm), positioning absent APs immediately below the hardware noise floor while maintaining monotonic physical ordering: strong signal ($-30$) $>$ moderate ($-70$) $>$ weak ($-100$) $>$ undetected ($-110$).
2. **Min-Max Scaling to $[0, 1]$:**
   $$x' = \frac{r - (-110)}{0 - (-110)}$$
   Here, undetected APs map to $0.0$, and the strongest detected signals approach $1.0$.
3. **Exponential Transformation (Positive Scaling):**
   $$x' = \begin{cases} \frac{\text{RSSI} - \text{RSSI}_{\min}}{\text{RSSI}_{\max} - \text{RSSI}_{\min}} & \text{if detected} \\ 0 & \text{if unobserved} \end{cases}$$
4. **Binary Presence Indicators:** Augmenting features with a binary flag vector $\mathbf{m} \in \{0, 1\}^{520}$ where $m_i = 1$ if $\text{WAP}_i$ is detected and $0$ otherwise.

### Q16: What is data leakage in this dataset, and how must metadata columns be treated?
**Answer:**
Attributes such as `SPACEID`, `RELATIVEPOSITION`, `USERID`, `PHONEID`, and `TIMESTAMP` provide operational context but present severe leakage risks:
- `USERID` / `PHONEID`: Certain users or devices surveyed only specific corridors or buildings during specific collection campaigns. If fed to a model, the algorithm learns spurious correlations (e.g., User 14 implies Building 2) rather than RF mapping, leading to near-zero training error but total failure on new devices.
- In fact, `SPACEID`, `RELATIVEPOSITION`, and `USERID` are deliberately zeroed out in the held-out validation set by the dataset creators.
- **Protocol:** Strip all metadata from model input matrices; train purely on the 520 WAP features.

### Q17: What is device heterogeneity and why does it degrade Wi-Fi positioning?
**Answer:**
Different smartphone vendors utilize different Wi-Fi chipsets, internal antenna orientations, casing attenuations, and driver calibration offsets. A Samsung phone and an HTC phone placed at the exact same physical coordinates may register RSSI discrepancies of $10\text{ to }20$ dBm for the same AP.
Because the validation set was gathered using different phone models and months later than the training set, models that overfit to specific absolute decibel thresholds suffer generalization loss. Multitask representations and normalization techniques help learn invariant spatial contours rather than absolute power levels.

---

## Part 5: Research Gap, Multitask Loss Balancing, and Limitations

### Q18: What is our specific research gap and scientific motivation?
**Answer:**
- **Status Quo in Literature:**
  1. Most classical ML papers (e.g., Torres-Sospedra et al., 2015; Salamah et al., 2016) treat building, floor, and coordinate prediction as isolated, decoupled tasks.
  2. Hierarchical deep architectures (e.g., Kim et al., 2018) often use sequential pipelines (predict building $\to$ select building-specific sub-network $\to$ predict floor $\to$ predict coordinates). Such cascading systems suffer from error propagation: an initial building misclassification irrevocably ruins downstream coordinate predictions.
- **Our Proposed Investigation:**
  A systematic, controlled benchmark on identical preprocessed inputs comparing:
  - Isolated classical baselines (KNN, Random Forest, XGBoost)
  - Single-task deep neural networks (STL-DNN)
  - Unified Hard-Parameter-Sharing Multitask Deep Learning (MTL-DNN) optimizing joint building, floor, and coordinate losses.
  We explicitly test Caruana's inductive transfer hypothesis: *Does learning coarse spatial hierarchies jointly as auxiliary tasks improve the fine-grained continuous coordinate regression accuracy?*

### Q19: How is the multitask loss function formulated and balanced?
**Answer:**
The combined multi-objective loss is defined as:
$$\mathcal{L}_{\text{total}} = \lambda_b \mathcal{L}_{\text{building}} + \lambda_f \mathcal{L}_{\text{floor}} + \lambda_c \mathcal{L}_{\text{coord}}$$
where:
- $\mathcal{L}_{\text{building}} = -\sum_{k=0}^{B-1} y_{b,k} \log \hat{y}_{b,k}$ (Categorical Cross-Entropy, 3 classes)
- $\mathcal{L}_{\text{floor}} = -\sum_{k=0}^{F-1} y_{f,k} \log \hat{y}_{f,k}$ (Categorical Cross-Entropy, 5 classes)
- $\mathcal{L}_{\text{coord}} = \frac{1}{2}\left(\|\hat{\mathbf{y}}_c - \mathbf{y}_c\|_2^2\right)$ or Smooth L1 / Huber Loss in normalized coordinate space.

**The Balancing Challenge:**
Cross-entropy losses typically range between $0.1$ and $1.5$, while unnormalized UTM coordinate MSE can be in thousands of square metres ($(\Delta x)^2 \sim 100\text{--}2500\text{ m}^2$).
Without normalization or loss weighting ($\lambda_b, \lambda_f, \lambda_c$), coordinate gradients dwarf classification gradients, causing negative transfer.
- **Remedy:** Normalize coordinates to $[0, 1]$ or zero-mean unit-variance, grid-search loss weights, or apply adaptive gradient balancing methods such as GradNorm (Chen et al., 2018).

### Q20: What are the primary threats to validity and limitations of our research design?
**Answer:**
1. **Single-Dataset Limitation:** Benchmarking is conducted on UJIIndoorLoc. While it is the de facto academic standard, findings should be validated on secondary datasets (e.g., Tampere, TUT) before making universal indoor propagation claims.
2. **Temporal Fading & Environment Dynamics:** Wi-Fi radio maps degrade over time as office furniture shifts, doors open/close, and human occupancy varies. Offline fingerprinting requires periodic calibration or semi-supervised radio-map updating.
3. **Computational Resource Trade-offs:** While neural inference on a forward pass takes under 1 ms, deep multitask networks require substantial hyperparameter tuning and backpropagation overhead compared to non-parametric KNN or off-the-shelf tree models.
