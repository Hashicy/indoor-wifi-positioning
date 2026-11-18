"""Generate the official final_presentation.pptx for Member 5 (12 slides).

Matches the exact slide-by-slide structure defined in the Project Guide:
Slide 1: Title and team members
Slide 2: Problem statement
Slide 3: Dataset overview
Slide 4: Wi-Fi fingerprinting explanation
Slide 5: Proposed architecture
Slide 6: Multitask neural network
Slide 7: Hybrid innovation
Slide 8: Loss functions and regularization
Slide 9: Experimental setup
Slide 10: Results and comparison
Slide 11: Limitations and future work
Slide 12: Conclusion
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    # 16:9 widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    # Color palette
    NAVY = RGBColor(26, 54, 93)       # #1A365D - Primary dark
    SLATE = RGBColor(71, 85, 105)     # #475569 - Secondary text
    LIGHT_BG = RGBColor(248, 250, 252)# #F8FAFC - Background card
    ACCENT_BLUE = RGBColor(37, 99, 235) # #2563EB - Highlights
    DARK_TEXT = RGBColor(15, 23, 42)  # #0F172A - Body text
    WHITE = RGBColor(255, 255, 255)
    BORDER_COLOR = RGBColor(226, 232, 240)
    
    blank_layout = prs.slide_layouts[6] # blank slide
    
    def add_header(slide, title_text, category_text="INDOOR POSITIONING RESEARCH PROJECT"):
        # Header banner card
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE
        
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.7))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = NAVY
        
    def add_card(slide, left, top, width, height, title="", fill_color=LIGHT_BG):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
        shape.line.color.rgb = BORDER_COLOR
        shape.line.width = Pt(1)
        
        if title:
            tx = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
            tf = tx.text_frame
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = NAVY
        return shape

    # =========================================================================
    # SLIDE 1: Title and team members
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()
    
    tbox = s1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11), Inches(2.2))
    tf1 = tbox.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "INDOOR WI-FI POSITIONING"
    p1.font.size = Pt(38)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    
    p2 = tf1.add_paragraph()
    p2.text = "Via Signal Strength Features, Deep Multitask Learning & Hybrid Ensembles"
    p2.font.size = Pt(22)
    p2.font.color.rgb = RGBColor(191, 219, 254)
    p2.space_before = Pt(12)
    
    # Team members card
    card_team = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(4.2), Inches(10.9), Inches(2.2))
    card_team.fill.solid()
    card_team.fill.fore_color.rgb = RGBColor(30, 41, 59)
    card_team.line.color.rgb = RGBColor(51, 65, 85)
    
    tf_team = card_team.text_frame
    tf_team.word_wrap = True
    p_t0 = tf_team.paragraphs[0]
    p_t0.text = "Project Team & Workstream Allocations (Five-Member DL Implementation)"
    p_t0.font.size = Pt(14)
    p_t0.font.bold = True
    p_t0.font.color.rgb = RGBColor(147, 197, 253)
    
    members = [
        "• Member 1: Deep Learning Architecture & Core Pipeline (Multitask Net, Data Split, Scalers)",
        "• Member 2: Hybrid Innovation (Out-of-Fold Tree Stacking & Residual GBDT Correction)",
        "• Member 3: Loss Functions, Regularization & Ablation Studies (Uncertainty Weighting, Noise)",
        "• Member 4: Comprehensive Evaluation, Diagnostics & Spatial Visualization (Metrics, CDF, Maps)",
        "• Member 5: Literature Review, Theoretical Foundations, System Integration & Final Report"
    ]
    for m in members:
        pm = tf_team.add_paragraph()
        pm.text = m
        pm.font.size = Pt(11)
        pm.font.color.rgb = RGBColor(241, 245, 249)
        pm.space_before = Pt(3)

    # =========================================================================
    # SLIDE 2: Problem statement
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement: The Challenge of Indoor Positioning")
    
    add_card(s2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3), "The GNSS / GPS Indoor Blindspot")
    tx = s2.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    bullets2_left = [
        "Severe Attenuation: Microwave satellite signals (~1.5 GHz) cannot penetrate reinforced concrete slabs, steel frames, and tinted glazing.",
        "Multipath Distortion: Non-line-of-sight reflections, diffraction, and dynamic scattering destroy geometric Time-of-Flight (ToF) assumptions.",
        "Ubiquitous Need: Humans spend >80% of daily life indoors; emergency navigation, airport logistics, smart hospitals require reliable positioning.",
        "Zero-Infrastructure Constraint: Dedicated systems (e.g., UWB) require costly infrastructure. Wi-Fi leverages existing IEEE 802.11 AP networks."
    ]
    for i, b in enumerate(bullets2_left):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {b}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)

    add_card(s2, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), "Three Fundamental Core Challenges")
    tx = s2.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    bullets2_right = [
        "1. Extreme Feature Sparsity (>96.5% Unobserved):\nOnly 15–18 of 520 access points are active per scan. Unobserved APs must be modeled without corrupting real signal decibels.",
        "2. Device Heterogeneity & Temporal Drift:\nAntenna gains, chipsets, and driver offsets vary up to 15 dBm across vendors; fingerprints degrade over time due to seasonal multipath changes.",
        "3. Coupled Hierarchical Spatial Targets:\nPositioning requires discrete topological classification (Building, Floor) AND continuous regression (UTM Coordinates x, y). Decoupled or cascading pipelines suffer catastrophic error compounding."
    ]
    for i, b in enumerate(bullets2_right):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = b
        p.font.size = Pt(12)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(10)

    # =========================================================================
    # SLIDE 3: Dataset overview
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Dataset Overview: UJIIndoorLoc Benchmark")
    
    add_card(s3, Inches(0.8), Inches(1.5), Inches(4.2), Inches(5.3), "Dataset Specification")
    tx = s3.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(3.8), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    d_specs = [
        "Benchmark: UJIIndoorLoc (UCI Repository)",
        "Environment: Universitat Jaume I, Spain (3 Buildings, 110,000 m²)",
        "Fingerprint Dimensions: 520 Wireless Access Points (WAP001–WAP520)",
        "Raw RSSI Range: -104 dBm to 0 dBm; sentinel +100 denotes not detected.",
        "Targets: Building (3), Floor (5, 13 valid pairs), Coordinates (UTM x, y).",
        "Total Records: 19,937 train + 1,111 validation (official test set)."
    ]
    for i, s in enumerate(d_specs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {s}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(6)

    add_card(s3, Inches(5.3), Inches(1.5), Inches(7.2), Inches(5.3), "Data Pipeline & Strict Leakage-Free Split")
    tx = s3.shapes.add_textbox(Inches(5.5), Inches(2.1), Inches(6.8), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    split_info = [
        "1. Cleaning Stage (Phase 1 Protocol Preserved):\n   Removed 637 duplicate rows and 76 all-sentinel scans -> 19,227 valid fingerprints.",
        "2. GroupShuffleSplit by Reference Position (15% Holdout):\n   Each spatial point was surveyed ~20 times. A naive random split leaks identical scans. Position-grouped splitting ensures train (16,432) and val (2,795) share zero positions.",
        "3. Strictly Untouched Test Set (validationData.csv):\n   Official 1,111 scans used exclusively for final evaluation. Features a 4-month temporal gap and 9 unseen smartphone models (true OOD test).",
        "4. Feature Filtering & Scaling:\n   Filtered to 459 APs observed in training. Sentinel (+100) mapped to -110 dBm (below floor), min-max scaled to [0, 1]. Coordinates standard-scaled."
    ]
    for i, s in enumerate(split_info):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = s
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 4: Wi-Fi fingerprinting explanation
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Wi-Fi Fingerprinting: Physical Foundations & Mechanics")
    
    add_card(s4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3), "The Fingerprinting Concept")
    tx = s4.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    fp_text = [
        "Empirical Radio Mapping:\nBypasses theoretical path-loss equations by mapping observed signal strength vectors directly to spatial coordinates.",
        "Offline Survey Phase:\nSurveyors walk the facility recording RSSI vectors x = [r_1, r_2, ..., r_520] along with ground truth (bldg, floor, UTM x, y).",
        "Online Positioning Phase:\nA mobile query fingerprint x* is fed to the trained inference engine, which simultaneously resolves building, floor, and (x, y) coordinates.",
        "Why Sentinel Preprocessing is Vital:\nRaw +100 acts as a massive +100 dBm transmitter in neural nets and distorts distances in KNN. Mapping +100 -> -110 dBm enforces physical monotonicity: strong (-30) > moderate (-70) > weak (-100) > absent (-110)."
    ]
    for i, t in enumerate(fp_text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)

    add_card(s4, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), "Classical Machine Learning Baselines")
    tx = s4.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    cml_text = [
        "Weighted k-Nearest Neighbors (WKNN):\nCanonical instance-based baseline. Predicts coordinates via inverse signal-space distance weighting. High memory footprint; O(M*N) test latency.",
        "Random Forest (Phase 1 Champion):\n200 decorrelated trees. Naturally isolates absent APs via axis-aligned splits. Achieved 99.73% building accuracy, 89.92% floor accuracy, and 12.30 m mean error.",
        "XGBoost (Gradient Boosted Trees):\nStrong classification performance (99.82% building) but struggles with continuous regression (22.70 m mean error) due to step-function quantization.",
        "The Step-Function Bottleneck:\nTrees divide continuous space into axis-aligned hyper-rectangles. Within a room, predictions are constant, bounding coordinate resolution."
    ]
    for i, t in enumerate(cml_text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 5: Proposed architecture
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Proposed Architecture: End-to-End System Pipeline")
    
    # 4 workflow step cards
    step_data = [
        ("1. Input & Cleaning", "520 Access Points\nReplace +100 with -110 dBm\nMinMax Scaler -> [0, 1]\nFilter to 459 active APs", Inches(0.8)),
        ("2. Shared Neural Trunk", "Dense 512 -> LN -> GELU\nDense 256 -> LN -> GELU\nDense 128 -> LN -> GELU\nDropout (p = 0.2-0.3)", Inches(3.8)),
        ("3. Task-Specific Heads", "Building Head: 3 logits\nFloor Head: 5 logits\nCoordinate Head: 2 coords\nSimultaneous multi-output", Inches(6.8)),
        ("4. Hybrid Enhancement", "OOF Tree Stacking (p_bld, p_flr)\nResidual GBDT Correction\nUncertainty Loss Weighting\nSub-ms inference (<0.8 ms)", Inches(9.8))
    ]
    for title, desc, left in step_data:
        add_card(s5, left, Inches(1.5), Inches(2.7), Inches(2.8), title)
        tx = s5.shapes.add_textbox(left + Inches(0.1), Inches(2.1), Inches(2.5), Inches(2.0))
        tf = tx.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT

    add_card(s5, Inches(0.8), Inches(4.6), Inches(11.7), Inches(2.2), "Key Architectural Principles")
    tx = s5.shapes.add_textbox(Inches(1.0), Inches(5.1), Inches(11.3), Inches(1.6))
    tf = tx.text_frame
    tf.word_wrap = True
    arch_pts = [
        "• Hard Parameter Sharing: Eliminates cascading error propagation. An early floor error does not crash coordinate regression.",
        "• Inductive Transfer (Caruana): Shared hidden layers learn common representations benefiting classification and continuous regression.",
        "• Layer Normalization + GELU: Provides smooth, continuous gradients across sparse inputs; avoids dead neurons common in standard ReLU.",
        "• Comprehensive Integration: Unifies classical ensemble reliability with deep continuous coordinate representation."
    ]
    for i, pt in enumerate(arch_pts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(2)

    # =========================================================================
    # SLIDE 6: Multitask neural network
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Multitask Neural Network: Layer Specifications & Multi-Head Outputs")
    
    add_card(s6, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3), "Network Layer Specifications")
    tx = s6.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    nn_specs = [
        "Input Layer: 459 min-max scaled RSSI features [0, 1]",
        "Block 1: Linear(459 -> 512) -> LayerNorm -> GELU -> Dropout(0.2)",
        "Block 2: Linear(512 -> 256) -> LayerNorm -> GELU -> Dropout(0.2)",
        "Block 3: Linear(256 -> 128) -> LayerNorm -> GELU (Shared Latent z)",
        "Total Model Parameters: ~400,000 trainable weights",
        "Optimizer: Adam (learning rate = 1e-3, batch size = 128)",
        "Selection Criterion: Minimized EvAAL score on position-grouped validation set with early stopping (patience = 20 epochs)."
    ]
    for i, s in enumerate(nn_specs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {s}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(6)

    add_card(s6, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), "Three Task-Specific Output Heads")
    tx = s6.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    heads_desc = [
        "1. Building Head (Discrete Multiclass):\n   Linear(128 -> 3). Raw logits passed to CrossEntropyLoss. Predicts Building 0, 1, or 2.",
        "2. Floor Head (Discrete Multiclass):\n   Linear(128 -> 5). Raw logits passed to CrossEntropyLoss. Predicts Floor 0, 1, 2, 3, or 4.",
        "3. Coordinate Head (Continuous Bivariate Regression):\n   Linear(128 -> 2). Evaluates Mean Squared Error on standardized (x, y) coordinates.",
        "Single-Task vs. Multitask Benchmark:\n   Multitask matches single-task floor accuracy (91.0% vs 91.9%) with 3x fewer total parameters and unified sub-millisecond inference."
    ]
    for i, h in enumerate(heads_desc):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = h
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 7: Hybrid innovation
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Hybrid Innovation: Stacking & Residual Coordinate Correction")
    
    add_card(s7, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3), "Approach A: Out-of-Fold Probability Stacking")
    tx = s7.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    stack_text = [
        "Motivation: Tree models excel at discrete threshold splits; neural nets excel at continuous coordinate manifolds.",
        "Random Forest Classifiers: 100-tree building and floor models generate class posterior probability vectors (3 bldg + 5 floor = 8 features).",
        "Strict Leakage-Free OOF Protocol:\n   Training probabilities are generated via 5-fold stratified cross-validation out-of-fold. The neural network never sees in-sample overconfident tree predictions.",
        "Stacked Representation:\n   z = [x_rssi, p_bld, p_flr] in R^467 fed to the deep multitask network.",
        "Empirical Gain: Floor accuracy rises from 91.00% to 92.08%."
    ]
    for i, t in enumerate(stack_text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(6)

    add_card(s7, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), "Approach B: Residual GBDT Correction")
    tx = s7.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    resid_text = [
        "Concept: Deep neural net predicts smooth global coordinates; systematic local distortions (wall attenuation, doorway bottlenecks) remain in residuals.",
        "Residual Target Calculation:\n   e_x = x_true - x_hat_DL\n   e_y = y_true - y_hat_DL",
        "Gradient Boosted Regressor:\n   XGBoost / LightGBM trained on RSSI features to predict spatial residuals (e_hat_x, e_hat_y).",
        "Final Hybrid Prediction:\n   x_final = x_hat_DL + e_hat_x\n   y_final = y_hat_DL + e_hat_y",
        "Combined Synergy: Full Hybrid (Stacking + Residual GBDT) drops mean error to 8.51 m, median error to 6.30 m, and EvAAL score to 8.96."
    ]
    for i, t in enumerate(resid_text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 8: Loss functions and regularization
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Loss Functions, Regularization & Ablation Studies")
    
    add_card(s8, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3), "Uncertainty-Based Loss Balancing")
    tx = s8.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    loss_text = [
        "The Loss Scale Dilemma:\n   L_total = L_bld + L_flr + L_coords. Cross-entropy losses (~0.1-1.5) dominate standardized coordinate MSE (~0.05), causing coordinate under-training.",
        "Homoscedastic Uncertainty Weighting (Kendall et al.):\n   L_total = sum_i [ exp(-s_i) * L_i + c_i * s_i ]\n   where s_i = log(sigma_i^2) is a learnable log-variance parameter.",
        "Likelihood Coefficients:\n   c_i = 1.0 for classification (Softmax likelihood)\n   c_i = 0.5 for regression (Gaussian likelihood)",
        "Dynamic Precision: When a loss is large, s_i increases to prevent task dominance, while c_i*s_i penalizes divergence."
    ]
    for i, t in enumerate(loss_text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)

    add_card(s8, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), "Regularization & Ablation Outcomes")
    tx = s8.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    ablation_text = [
        "Run A (Baseline, p=0.2): Bld 99.73%, Flr 91.00%, Mean 8.95 m (EvAAL 9.496)",
        "Run B (Uncertainty Weighting): Bld 99.82%, Flr 91.09%, Mean 9.19 m",
        "Run C (No Dropout, p=0.0): Bld 99.55%, Flr 91.00%, Mean 9.49 m (Overfitting)",
        "Run D (Dropout p=0.3): Bld 99.73%, Flr 91.81%, Mean 8.85 m (EvAAL 9.359 - Best Single Model)",
        "Run E (RSSI Noise, sigma=0.02): Bld 99.91%, Flr 90.91%, Mean 8.97 m",
        "Run F (AP Dropout, p=0.10): Bld 99.91%, Flr 90.19%, Mean 9.05 m",
        "Run G (Full Combo): Bld 99.73%, Flr 90.55%, Mean 9.01 m",
        "Key Finding: Dropout = 0.3 strongly suppresses co-adaptation and achieves best generalization on the 4-month delayed test set."
    ]
    for i, t in enumerate(ablation_text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(10)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(4)

    # =========================================================================
    # SLIDE 9: Experimental setup
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Experimental Setup & Evaluation Protocol")
    
    add_card(s9, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3), "Standardized Evaluation Protocol")
    tx = s9.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    setup_text = [
        "Held-Out Benchmark Test Set:\n   Official validationData.csv (1,111 scans). Kept untouched until final evaluation. No hyperparameter tuning on test.",
        "Reproducibility Controls:\n   Fixed random seeds (42, 7, 123) for cross-seed stability checks. Train/val/test splits strictly identical across all members.",
        "Verified Metric Units:\n   UTM coordinates are in meters. Positioning errors are reported in real Euclidean meters (not scaled units).",
        "Hardware & Execution:\n   PyTorch 2.15, Python 3.14 on macOS Apple Silicon. Training time per model: 30-50s on CPU; test inference latency: <0.8 ms."
    ]
    for i, t in enumerate(setup_text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {t}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)

    add_card(s9, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), "Mathematical Metrics Formulations")
    tx = s9.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    metrics_math = [
        "1. Classification Accuracy:\n   Acc = (1/N) sum I(y_hat_i == y_i), evaluated for Building, Floor, and Joint (both correct).",
        "2. Euclidean Positioning Error (2D Metric Space):\n   d_i = sqrt((x_i - x_hat_i)^2 + (y_i - y_hat_i)^2)\n   Reported as Mean, Median (P50), P75, P90, and Maximum.",
        "3. Official EvAAL Competition Score:\n   Score = Mean(d_i) + 50 * (1 - Acc_bld) + 4 * Mean(|f_i - f_hat_i|)\n   Explicitly penalizes severe building breaches (50m) and floor errors (4m/floor)."
    ]
    for i, t in enumerate(metrics_math):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(8)

    # =========================================================================
    # SLIDE 10: Results and comparison
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Experimental Results & Cross-Model Benchmark")
    
    # Summary Table
    rows = [
        ["Model Architecture", "Paradigm", "Bld Acc", "Flr Acc", "Mean Err", "Med Err", "P90 Err", "EvAAL"],
        ["Random Forest (Baseline)", "Classical", "99.73%", "89.92%", "12.30 m", "8.15 m", "29.55 m", "13.04"],
        ["k-NN (k=5, weighted)", "Classical", "98.92%", "75.43%", "13.68 m", "8.20 m", "29.21 m", "15.54"],
        ["XGBoost", "Classical", "99.82%", "89.56%", "22.70 m", "15.17 m", "47.77 m", "23.41"],
        ["Single-Task DL Models", "Deep Learning", "100.00%", "91.90%", "8.66 m", "6.37 m", "17.13 m", "9.02"],
        ["Multitask DL (Shared)", "Deep Learning", "99.73%", "91.00%", "8.95 m", "7.04 m", "17.87 m", "9.50"],
        ["Hybrid: Tree Stacking", "Hybrid", "99.82%", "92.08%", "8.86 m", "6.69 m", "17.72 m", "9.31"],
        ["Hybrid: Residual GBDT", "Hybrid", "99.73%", "91.00%", "8.75 m", "6.69 m", "17.59 m", "9.29"],
        ["Full Hybrid (Stack + GBDT)", "Hybrid", "99.82%", "92.08%", "8.51 m", "6.30 m", "17.58 m", "8.96"]
    ]
    
    table_shape = s10.shapes.add_table(len(rows), len(rows[0]), Inches(0.8), Inches(1.5), Inches(11.7), Inches(3.2))
    table = table_shape.table
    
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(10)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            elif r_idx == len(rows) - 1:
                p.font.bold = True
                p.font.color.rgb = NAVY
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(220, 252, 231)
            else:
                p.font.color.rgb = DARK_TEXT
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 1 else RGBColor(248, 250, 252)

    add_card(s10, Inches(0.8), Inches(4.9), Inches(11.7), Inches(2.0), "Key Benchmark Takeaways")
    tx = s10.shapes.add_textbox(Inches(1.0), Inches(5.3), Inches(11.3), Inches(1.5))
    tf = tx.text_frame
    tf.word_wrap = True
    takeaways = [
        "• Deep Learning Outperforms Classical ML: Multitask DL reduces mean positioning error by 27.2% vs Random Forest (8.95 m vs 12.30 m).",
        "• Full Hybrid Achieves State-of-the-Art: Combining tree structural priors with residual boosting drops mean error to 8.51 m, median to 6.30 m, and EvAAL score to 8.96.",
        "• Floor Error Diagnostics: 90.0% of floor mistakes are off by exactly 1 adjacent floor (RF ceiling slab attenuation of 10-18 dB).",
        "• Bound on Worst-Case Uncertainty: Full Hybrid compresses P90 error from 29.55 m (RF) to 17.58 m (a 40.5% reduction in tail error)."
    ]
    for i, t in enumerate(takeaways):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = t
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(2)

    # =========================================================================
    # SLIDE 11: Limitations and future work
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Limitations, Threats to Validity & Future Directions")
    
    add_card(s11, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3), "Threats to Validity & Limitations")
    tx = s11.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.2), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    limits = [
        "1. Single Campus Corpus (UJIIndoorLoc):\n   Evaluated on three buildings in Spain. Architectural features (atrium voids, ceiling heights) are site-specific. Needs cross-validation on TUT/Tampere.",
        "2. Four-Month Temporal Drift:\n   Train (May-Jun 2013) vs Test (Sep-Oct 2013). Changing human occupancy, office layouts, and foliage cause temporal radio-map degradation.",
        "3. Smartphone Hardware Heterogeneity:\n   16 devices in train vs 11 in test (9 unobserved). Antenna gain offsets (up to 15 dBm) cause spatial boundary shifts.",
        "4. Geometric Dilution of Precision (DoP):\n   Scans with <= 8 detected APs exhibit higher errors (mean 10.89 m); signal contour intersections become ill-conditioned."
    ]
    for i, l in enumerate(limits):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = l
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(6)

    add_card(s11, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3), "Future Research Roadmaps")
    tx = s11.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.5))
    tf = tx.text_frame
    tf.word_wrap = True
    future = [
        "1. Temporal & Kinematic Tracking:\n   Integrate sequence modeling (LSTM, GRU, Transformers) to incorporate user trajectories and floor transition constraints (stairs, elevators).",
        "2. Self-Supervised Domain Adaptation:\n   Contrastive pre-training to map RSSI vectors from heterogeneous smartphone chipsets into a canonical, hardware-invariant latent space.",
        "3. Semi-Supervised Continual Maintenance:\n   Employ passive user crowdsourcing to incrementally calibrate AP weightings without requiring labor-intensive site re-surveys.",
        "4. Graph Neural Networks (GNNs):\n   Explicitly model architectural topology (doors, hallways, rooms) as structural adjacency priors in deep representations."
    ]
    for i, f in enumerate(future):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f
        p.font.size = Pt(11)
        p.font.color.rgb = DARK_TEXT
        p.space_after = Pt(6)

    # =========================================================================
    # SLIDE 12: Conclusion
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    bg12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = NAVY
    bg12.line.fill.background()
    
    tbox12 = s12.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(1.2))
    tf12 = tbox12.text_frame
    tf12.word_wrap = True
    p12 = tf12.paragraphs[0]
    p12.text = "Summary & Final Conclusion"
    p12.font.size = Pt(28)
    p12.font.bold = True
    p12.font.color.rgb = WHITE
    
    c_final = s12.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.0), Inches(11.3), Inches(4.8))
    c_final.fill.solid()
    c_final.fill.fore_color.rgb = RGBColor(30, 41, 59)
    c_final.line.color.rgb = RGBColor(51, 65, 85)
    
    tff = c_final.text_frame
    tff.word_wrap = True
    concl_points = [
        "1. Complete Five-Member Implementation:\n   From EDA and preprocessing to unified multitask deep learning, hybrid ensembles, loss balancing, error analysis, and reproducibility validation.",
        "2. Definite Superiority of Continuous Neural Manifolds:\n   Multitask learning successfully unifies building, floor, and coordinate estimation into a single model, overcoming the discrete step-function limits of tree models and dropping mean error to 8.95 m.",
        "3. Power of Hybrid Synergy:\n   The Full Hybrid model (Stacking + Residual GBDT) established new project benchmarks on the untouched test set:\n   -> Building Accuracy: 99.82%  |  Floor Accuracy: 92.08%  |  Joint Accuracy: 92.08%\n   -> Mean Error: 8.51 m  |  Median Error: 6.30 m  |  P90 Error: 17.58 m  |  EvAAL Score: 8.96",
        "4. Methodological Rigor & Reproducibility:\n   Zero validation leakage, position-grouped splitting, 63 automated unit tests, clean modular codebase, and verified IEEE research report.",
        "5. Ready for Real-World Deployment:\n   Sub-millisecond inference latency (<0.8 ms) enables real-time edge execution on mobile devices."
    ]
    for i, cp in enumerate(concl_points):
        p = tff.paragraphs[0] if i == 0 else tff.add_paragraph()
        p.text = cp
        p.font.size = Pt(12)
        p.font.color.rgb = RGBColor(241, 245, 249)
        p.space_after = Pt(8)

    out_dir = "/Users/kchhillar13/me/indoor-wifi-positioning/presentation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "final_presentation.pptx")
    prs.save(out_path)
    print(f"Presentation saved successfully to: {out_path}")

if __name__ == "__main__":
    create_presentation()
