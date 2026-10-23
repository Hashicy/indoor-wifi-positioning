# Literature Review: Indoor Wi-Fi Positioning Using RSSI Features, Classical Machine Learning, and Deep Multitask Learning

> **Scope.** This review covers the theoretical and empirical foundations for our project. It is organised into nine subsections progressing from broad context (indoor positioning) through specific methods (KNN, ensemble learning, deep multitask learning) to the research gap, problem formulation, and research questions that motivate our experimental work.
>
> **Citation convention.** All citations use author–year format keyed to `references.bib`. Where a fact requires further primary-source verification it is marked **[NEEDS VERIFICATION]**.

---

## 1. Introduction to Indoor Positioning

The demand for location-aware services—navigation, asset tracking, emergency response—has driven extensive research into automatic user localisation (Torres-Sospedra et al., 2014). Outdoors, the Global Positioning System (GPS) achieves sub-metre accuracy under open-sky conditions. Indoors, however, GPS signals are severely attenuated by walls, ceilings, and structural materials, rendering satellite-based positioning unreliable (Torres-Sospedra et al., 2014). Indoor Positioning Systems (IPS) therefore rely on alternative sensing modalities, including Bluetooth Low Energy, ultra-wideband, inertial measurement units, and—most prevalently—Wireless Local Area Network (WLAN) infrastructure.

WLAN-based methods are attractive because Wi-Fi access points (APs) are already deployed in most modern buildings, eliminating the need for dedicated hardware. Among WLAN-based techniques, *fingerprinting* has emerged as the dominant paradigm: rather than modelling radio propagation physics, it treats localisation as a pattern-recognition problem in which observed signal patterns are matched to a pre-recorded database of location-labelled measurements (Torres-Sospedra et al., 2015).

---

## 2. Wi-Fi RSSI Fingerprinting

A Wi-Fi fingerprint is a vector of Received Signal Strength Indicator (RSSI) values captured at a given location from all detectable APs. Fingerprinting operates in two phases:

1. **Offline (survey) phase.** A radio map is constructed by recording RSSI vectors at known reference points throughout the target environment. Each record associates a vector of signal strengths with spatial coordinates and categorical labels (building, floor).

2. **Online (localisation) phase.** A user's device captures a live RSSI vector, which is compared against the radio map using a distance metric or a trained model to estimate the user's position.

The UJIIndoorLoc database (Torres-Sospedra et al., 2014) is a widely adopted benchmark for fingerprint-based indoor localisation. It spans three buildings at Universitat Jaume I covering approximately 110,000 m², with up to five floors per building. Each fingerprint contains RSSI readings from 520 detected WAPs, encoded as integers from −104 dBm (very weak) to 0 dBm, with the sentinel value +100 indicating that a WAP was not detected. The dataset comprises 19,937 training records and 1,111 validation records collected by more than 20 users across 25 different Android devices, deliberately introducing device heterogeneity (Torres-Sospedra et al., 2014). According to the UCI Machine Learning Repository documentation, the dataset contains no missing values and supports both classification (building, floor) and regression (longitude, latitude) tasks.

**A note on evaluation metrics.** Two fundamentally different metrics are standard in this domain, reflecting the two task types:

- *Classification accuracy* (also called *hit rate*) measures the fraction of test samples for which the predicted categorical label (building or floor) exactly matches the ground truth. An incorrect building or floor prediction is a discrete error with no notion of "how wrong" the prediction is.

- *Mean positioning error* (MPE) measures the average Euclidean distance in metres between predicted and true coordinates. Unlike accuracy, MPE is a continuous metric that captures the magnitude of positional deviation. Because UJIIndoorLoc uses a local coordinate system rather than geographic degrees, Euclidean distance is the appropriate measure.

These metrics are complementary: a model may achieve high floor accuracy while still exhibiting large coordinate errors within the correct floor, or vice versa. Both must be reported for a complete evaluation.

---

## 3. KNN, WKNN, and Instance-Based Methods

The K-Nearest Neighbors (KNN) algorithm is the most established baseline for fingerprint matching. Given a query fingerprint, KNN computes the distance (typically Euclidean) between the query and every fingerprint in the radio map, selects the *K* closest matches, and estimates the position as the arithmetic mean of their known coordinates (Torres-Sospedra et al., 2015).

Weighted KNN (WKNN) refines this approach by assigning higher weights to nearer neighbours, typically inversely proportional to their distance in signal space. Torres-Sospedra et al. (2015) conducted a comprehensive analysis of 21 distance and similarity measures for KNN and WKNN on the UJIIndoorLoc dataset. Their analysis confirmed that WKNN consistently outperforms unweighted KNN and that, while Euclidean distance remains a competitive default, alternative measures such as Sørensen distance can yield improvements depending on the prediction target. However, that study evaluated only KNN/WKNN variants and did not compare against ensemble or deep-learning methods.

**Strengths.** KNN is non-parametric, requires no training phase, and is simple to implement. **Limitations.** It scales poorly with radio-map size (distance must be computed to every reference point), is sensitive to the choice of *K* and distance metric, and does not learn abstract features from the data.

---

## 4. Random Forest and Tree-Based Ensemble Methods

Random Forest (RF) constructs an ensemble of decision trees, each trained on a bootstrapped subset of the data with random feature sampling (bagging). For classification, it uses majority voting; for regression, it averages predictions. Tree-based ensembles offer several properties well suited to RSSI data:

- **Robustness to noise.** Indoor RSSI values fluctuate due to multipath propagation, human movement, and device heterogeneity. Ensemble averaging mitigates the effect of individual noisy measurements.
- **Native handling of the undetected-WAP sentinel.** Trees can learn decision boundaries that naturally partition the +100 sentinel from actual signal values.
- **Feature importance.** RF provides built-in ranking of WAP importance, aiding interpretability and feature selection.

Salamah et al. (2016) applied Random Forest, Support Vector Machine, KNN, and Decision Tree classifiers to WiFi fingerprint-based room/zone classification, combined with Principal Component Analysis (PCA) for dimensionality reduction. They reported that PCA reduced computation time by up to 70% when paired with RF, and that tree-based classifiers were robust to RSSI noise in both static and dynamic scenarios. **[NEEDS VERIFICATION: exact reported accuracy numbers; their experiments used a custom dataset, not UJIIndoorLoc.]**

While RF has been applied to individual classification or regression tasks in indoor positioning, its use within a unified multi-target framework on UJIIndoorLoc remains limited in the literature we have reviewed.

---

## 5. Gradient Boosting and XGBoost

Whereas RF builds trees independently (bagging), gradient boosting constructs trees sequentially, with each new tree specifically targeting the residual errors of the preceding ensemble. XGBoost (Extreme Gradient Boosting) is an optimised implementation of gradient boosting that incorporates L1/L2 regularisation, efficient handling of sparse features, and parallelised tree construction.

XGBoost is a strong candidate for RSSI-based localisation because:

- RSSI vectors are inherently tabular and high-dimensional—exactly the data regime where gradient-boosted trees consistently achieve state-of-the-art results.
- Built-in regularisation penalises overly complex trees, reducing overfitting on noisy indoor signal measurements.
- Sparse-aware splitting naturally handles the large proportion of +100 (undetected) values in UJIIndoorLoc fingerprints.

To our knowledge, systematic comparisons of XGBoost against KNN, RF, and deep-learning baselines on UJIIndoorLoc's full set of prediction targets (building, floor, longitude, latitude) are not prominent in the published literature we have reviewed. **[NEEDS VERIFICATION: a broader literature search may reveal such comparisons.]** Our project proposes to include XGBoost as a classical ML baseline alongside KNN/WKNN and RF.

---

## 6. Neural Networks and Multitask Learning

### 6.1 Deep Neural Networks for Indoor Localisation

Several studies have applied deep learning to WiFi fingerprint-based localisation on UJIIndoorLoc:

- **Nowicki and Wietrzykowski (2017)** demonstrated that a deep neural network (DNN) classifier achieves high building and floor recognition accuracy with minimal feature engineering. Their work confirmed the viability of DNNs for RSSI-based classification but did not extend to coordinate regression.

- **Kim, Lee, and Huang (2018)** proposed a scalable DNN architecture that uses stacked autoencoders (SAE) for feature extraction followed by a feed-forward classifier for hierarchical multi-building and multi-floor localisation. Their unified architecture handles both building and floor prediction and achieved near-state-of-the-art performance on UJIIndoorLoc with reduced computational cost. **[NEEDS VERIFICATION: whether their architecture uses formal multitask learning with a combined loss function, or sequential hierarchical classification with separate losses.]**

- **Song et al. (2019)** introduced CNNLoc, a convolutional neural network framework that treats RSSI vectors as one-dimensional signals and applies 1D convolutions to extract local patterns among WAP clusters. CNNLoc achieved competitive results on UJIIndoorLoc for building/floor classification and coordinate prediction. **[NEEDS VERIFICATION: whether their architecture employs multitask output heads or separate models for each target.]**

- **Feng, Nguyen, and Luo (2022)** surveyed deep learning approaches for WiFi-based indoor positioning, categorising methods by architecture type (DNN, CNN, RNN, autoencoder, GAN) and noting a trend toward hierarchical and multi-output models. Their survey identifies UJIIndoorLoc as the dominant benchmark dataset in the field.

### 6.2 Multitask Learning: Theoretical Foundations

Multitask learning (MTL) is an inductive transfer technique in which a single model learns multiple related tasks simultaneously through a shared representation (Caruana, 1997). The foundational insight is that training signals from auxiliary tasks constrain the shared representation to prefer hypotheses that generalise across all tasks, rather than overfitting to any single one.

Caruana (1997) identified five mechanisms by which MTL improves generalisation:

1. **Data amplification.** Gradient signals from all tasks jointly train the shared layers, effectively increasing the training signal for the common representation.
2. **Attention focusing.** Auxiliary tasks help the model attend to features relevant to the main task by providing additional gradient pressure.
3. **Eavesdropping.** Features that are difficult for one task to learn may be easily learned by another; sharing representations allows cross-task knowledge transfer.
4. **Representation bias.** MTL favours simpler, more general representations that serve multiple purposes—a form of Occam's razor.
5. **Regularisation.** Joint training acts as a regulariser, reducing overfitting by constraining the hypothesis space.

The most common MTL architecture is *hard parameter sharing* (Ruder, 2017): all tasks share a common set of hidden layers (the encoder), while each task has its own output head. Ruder (2017) surveyed MTL in deep neural networks, categorising architectures into hard and soft parameter sharing, cross-stitch networks, and sluice networks, and discussed practical considerations such as auxiliary task selection and loss weighting.

### 6.3 Loss Balancing in Multitask Learning

When MTL combines tasks with different loss types and scales—for example, cross-entropy for classification and mean squared error for regression—naive equal weighting can cause one task's gradients to dominate training, leading to poor performance on the other tasks. Chen et al. (2018) proposed GradNorm, a gradient normalisation method that dynamically adjusts task loss weights during training to equalise gradient norms across tasks. GradNorm was shown to outperform both fixed weighting and uncertainty-based weighting on several multi-task benchmarks. **Note:** GradNorm was evaluated on computer-vision tasks; its applicability to WiFi localisation has not been demonstrated in the original paper and would need to be validated experimentally.

### 6.4 Applying MTL to Indoor Localisation

Indoor WiFi positioning on UJIIndoorLoc is a natural candidate for MTL because the prediction targets are inherently related:

- Building identity, floor level, and planar coordinates all describe different aspects of the same underlying physical location.
- Building and floor classification can be viewed as coarse-grained localisation that constrains the coordinate regression to a specific spatial partition.
- Conversely, coordinate patterns may help disambiguate floor boundaries where RSSI signatures overlap between adjacent floors.

In a hard-parameter-sharing architecture applied to this problem, a shared encoder maps a preprocessed RSSI fingerprint to a latent representation, which then feeds into three task-specific heads: a softmax head for building classification (3 classes), a softmax head for floor classification (5 classes), and a linear head for coordinate regression (2 outputs). The total loss is a weighted sum of the individual task losses:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{building}} + \beta \cdot \mathcal{L}_{\text{floor}} + \gamma \cdot \mathcal{L}_{\text{coords}}$$

where $\mathcal{L}_{\text{building}}$ and $\mathcal{L}_{\text{floor}}$ are categorical cross-entropy losses and $\mathcal{L}_{\text{coords}}$ is mean squared error. The weights $\alpha$, $\beta$, $\gamma$ may be set manually, tuned via grid search, or adapted dynamically using methods such as GradNorm (Chen et al., 2018).

---

## 7. Comparative Analysis of Existing Research

Table 1 summarises the reviewed literature along key dimensions relevant to our project.

| Study | Year | Model(s) | Dataset | Classification | Regression | MTL |
|-------|------|----------|---------|----------------|------------|-----|
| Torres-Sospedra et al. | 2014 | 1-NN baseline | UJIIndoorLoc | Building, floor | Coordinates | No |
| Torres-Sospedra et al. | 2015 | KNN/WKNN variants | UJIIndoorLoc | Building, floor | Coordinates | No |
| Salamah et al. | 2016 | RF, SVM, KNN, DT | Custom | Zone/room | No | No |
| Nowicki & Wietrzykowski | 2017 | DNN | UJIIndoorLoc | Building, floor | No | No |
| Kim et al. | 2018 | SAE + DNN | UJIIndoorLoc | Building, floor | Coordinates | Hierarchical* |
| Song et al. | 2019 | CNN (CNNLoc) | UJIIndoorLoc + others | Building, floor | Coordinates | Unclear* |
| Feng et al. | 2022 | Survey | Multiple | — | — | — |

*\* Whether Kim et al. (2018) and Song et al. (2019) employ formal multitask learning with a combined loss function, as opposed to sequential or separate prediction, requires direct confirmation from the full paper texts.*

Several patterns emerge from this comparison:

1. **Task separation.** Most studies address either classification (building/floor) or regression (coordinates) as their primary objective, with few explicitly optimising both through a unified loss.
2. **Method isolation.** Classical ML and deep learning approaches are typically compared within their own families; systematic cross-paradigm comparisons using identical preprocessing and evaluation protocols on UJIIndoorLoc are uncommon in the works we reviewed.
3. **Limited explicit MTL.** While hierarchical architectures (e.g., Kim et al., 2018) capture task dependencies implicitly, formal multitask learning with a shared encoder and combined classification-plus-regression loss—as described by Caruana (1997) and Ruder (2017)—has not been prominently applied to UJIIndoorLoc in the literature we surveyed. **[NEEDS VERIFICATION: a broader search may reveal recent studies addressing this gap.]**

---

## 8. Research Gap and Motivation

Based on the reviewed literature, we identify the following **hypothesised gap**, which requires validation against additional sources before any firm claim of novelty can be made:

> **Hypothesis.** Existing studies on WiFi fingerprint-based indoor localisation on UJIIndoorLoc predominantly treat prediction targets independently or in sequential hierarchical pipelines. Relatively few works have (a) systematically compared classical ML baselines (KNN/WKNN, Random Forest, XGBoost) against deep learning models under identical preprocessing, and (b) evaluated whether a formally defined multitask deep-learning model—jointly optimising building classification, floor classification, and coordinate regression through a single shared representation and combined loss—improves positioning accuracy compared to single-task alternatives.

This gap is motivated by Caruana's (1997) theoretical argument that related tasks benefit from shared representations, combined with the practical observation that building and floor labels are structurally related to coordinate positions. If building/floor classification acts as an effective auxiliary signal for coordinate regression (via the eavesdropping and attention-focusing mechanisms described by Caruana), then a multitask architecture should achieve lower mean positioning error than an equivalent single-task regression model trained on coordinates alone.

**Important caveats.** We do not claim that multitask learning has never been applied to indoor localisation; hierarchical and multi-output architectures exist in the literature (Kim et al., 2018; Song et al., 2019; Feng et al., 2022). Our claim is narrower: that a specific experimental comparison—classical ML baselines versus single-task DNN versus formally defined MTL DNN, evaluated on all four prediction targets under a common pipeline on UJIIndoorLoc—appears underrepresented in the works we have reviewed. This claim must be verified against a broader literature search before it can be stated in a published report.

---

## 9. Problem Formulation, Research Questions, and Expected Contributions

### 9.1 Problem Formulation

We formulate indoor WiFi positioning as three concurrent prediction tasks over the same input:

Let $\mathbf{x} = [r_1, r_2, \ldots, r_{520}] \in \mathbb{R}^{520}$ denote a preprocessed WiFi fingerprint. The targets are:

- **Building classification:** $y_b \in \{0, 1, 2\}$ — a 3-class classification problem with cross-entropy loss.
- **Floor classification:** $y_f \in \{0, 1, 2, 3, 4\}$ — a 5-class classification problem with cross-entropy loss.
- **Coordinate regression:** $\mathbf{y}_c = (\text{lon}, \text{lat}) \in \mathbb{R}^2$ — a two-output regression problem with mean squared error loss.

Building and floor are formulated as classification rather than regression because they are categorical labels: BuildingID encodes no ordinal distance (building 1 is not "between" buildings 0 and 2), and while floors are ordinal, their RSSI signatures do not vary linearly with floor number.

In the multitask formulation, a shared encoder $h_{\theta_s}$ maps the fingerprint to a latent representation $\mathbf{z}$, which is then consumed by three task-specific heads. All parameters are trained jointly by minimising $\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{building}} + \beta \cdot \mathcal{L}_{\text{floor}} + \gamma \cdot \mathcal{L}_{\text{coords}}$, where gradients from all tasks flow through the shared encoder—the mechanism by which MTL is hypothesised to produce richer representations (Caruana, 1997).

### 9.2 Research Questions

1. **RQ1.** How do classical ML models (KNN, WKNN, Random Forest, XGBoost) compare against deep-learning models for building classification, floor classification, and coordinate regression on the UJIIndoorLoc dataset?

2. **RQ2.** Does a multitask deep-learning model with a shared representation and combined classification-plus-regression loss achieve lower mean positioning error than an equivalent single-task deep regression model?

3. **RQ3.** How does the choice of loss-weighting strategy ($\alpha$, $\beta$, $\gamma$) affect multitask model performance across all prediction targets?

4. **RQ4.** What RSSI preprocessing and feature-selection strategies yield the best performance for both classical ML and deep-learning approaches?

### 9.3 Objectives

1. Implement and evaluate classical ML baselines (KNN, WKNN, Random Forest, XGBoost) for all four prediction targets on UJIIndoorLoc.
2. Train single-task deep neural networks for building classification, floor classification, and coordinate regression independently.
3. Design and train a multitask DNN with hard parameter sharing that jointly predicts building, floor, and coordinates.
4. Systematically compare all models using building accuracy, floor accuracy, and mean positioning error under identical preprocessing.
5. Investigate loss-weighting strategies for the multitask model.

### 9.4 Expected Contributions

The expected contributions are framed as *proposed experimental outcomes*, not as established results:

- A reproducible benchmark comparing classical ML and deep-learning approaches on UJIIndoorLoc's full prediction-target set under a unified evaluation protocol.
- An empirical evaluation of whether MTL with building/floor classification as auxiliary tasks improves coordinate regression accuracy, testing Caruana's (1997) theoretical predictions in the WiFi localisation domain.
- An analysis of loss-weighting strategies for combining classification and regression objectives in a single MTL model.

**We do not claim that our model outperforms existing methods.** Performance comparisons will be determined by the experimental results, which have not yet been conducted.

---

## References

All references cited in this review are maintained in [`references.bib`](file:///Users/shreshthagupta2708/Desktop/indoor-wifi-positioning/report/references.bib) and correspond to verified entries in [`literature_review_matrix.csv`](file:///Users/shreshthagupta2708/Desktop/indoor-wifi-positioning/research/literature_review_matrix.csv).

| Citation Key | Full Reference |
|---|---|
| Torres-Sospedra et al., 2014 | Torres-Sospedra, J. et al. "UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems." IPIN 2014. DOI: 10.1109/IPIN.2014.7275492 |
| Caruana, 1997 | Caruana, R. "Multitask Learning." *Machine Learning*, 28(1), 41–75. DOI: 10.1023/A:1007379606734 |
| Torres-Sospedra et al., 2015 | Torres-Sospedra, J. et al. "Comprehensive Analysis of Distance and Similarity Measures for Wi-Fi Fingerprinting Indoor Positioning Systems." *Expert Systems with Applications*, 42(23). DOI: 10.1016/j.eswa.2015.08.013 |
| Kim et al., 2018 | Kim, K.S., Lee, S., Huang, K. "A Scalable Deep Neural Network Architecture for Multi-Building and Multi-Floor Indoor Localization Based on Wi-Fi Fingerprinting." *Big Data Analytics*, 3(4). DOI: 10.1186/s41044-018-0031-2 |
| Nowicki & Wietrzykowski, 2017 | Nowicki, M., Wietrzykowski, J. "Low-Effort Place Recognition with WiFi Fingerprints Using Deep Learning." *Automation 2017*, Springer. DOI: 10.1007/978-3-319-54042-9_57 |
| Ruder, 2017 | Ruder, S. "An Overview of Multi-Task Learning in Deep Neural Networks." arXiv:1706.05098 |
| Chen et al., 2018 | Chen, Z. et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." ICML 2018. arXiv:1711.02257 |
| Song et al., 2019 | Song, X. et al. "A Novel Convolutional Neural Network Based Indoor Localization Framework With WiFi Fingerprinting." *IEEE Access*, 7, 110698–110709. DOI: 10.1109/ACCESS.2019.2933921 |
| Salamah et al., 2016 | Salamah, A.H. et al. "An Enhanced WiFi Indoor Localization System Based on Machine Learning." IPIN 2016. **[DOI pending manual lookup]** |
| Feng et al., 2022 | Feng, X., Nguyen, K.A., Luo, Z. "A Survey of Deep Learning Approaches for WiFi-Based Indoor Positioning." *J. Information and Telecommunication*, 6(2), 163–216. DOI: 10.1080/24751839.2021.1975425 |
