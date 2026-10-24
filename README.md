# Indoor Wi-Fi Positioning

Indoor Wi-Fi positioning using RSSI signal features, classical machine learning, and deep multitask learning to predict building, floor, and continuous metric coordinates with the UJIIndoorLoc benchmark dataset.

---

## Project Structure

```
indoor-wifi-positioning/
├── data/                      # Dataset files (UJIIndoorLoc train and validation sets)
├── models/                    # Saved model weights and checkpoints
├── notebooks/                 # Exploratory data analysis and experimental notebooks
├── presentation/              # Presentation slides and visual assets
├── report/                    # Project report (LaTeX / IEEEtran) & evaluation summaries
│   ├── figures/               # Report figures
│   ├── model_evaluation/      # Consolidated model comparison tables and evaluation reports
│   ├── sections/              # Modular LaTeX sections
│   │   ├── dataset_eda.tex
│   │   ├── feature_engineering.tex
│   │   ├── introduction.tex
│   │   ├── related_work.tex
│   │   └── traditional_ml.tex
│   ├── main.tex               # Master LaTeX document
│   └── references.bib         # Verified BibTeX references
├── research/                  # Member 1 Research & Literature Review
│   ├── literature_review_draft.md    # Full comprehensive literature review
│   ├── literature_review_matrix.csv  # 10-paper comparative analysis matrix
│   ├── paper_notes/                  # Deep-dive study notes
│   │   ├── caruana_1997.md           # Caruana (1997) Multitask Learning
│   │   └── ujiindoorloc_2014.md      # Torres-Sospedra et al. (2014) UJIIndoorLoc
│   ├── problem_formulation.md        # Mathematical problem formulation
│   ├── research_gap.md               # Empirical research gap and hypotheses
│   └── viva_questions.md             # 20 Viva Voce questions & model answers
├── results/                   # Evaluation results, logs, and generated plots
├── scripts/                   # Training and pipeline execution scripts
├── src/                       # Core source code (preprocessing, models, evaluation)
├── tests/                     # Unit and integration tests
└── requirements.txt           # Python project dependencies
```

---

## Member Responsibilities

- **Member 1 (Current Deliverables):** Literature review, research gap, theoretical foundation (Caruana MTL), mathematical problem formulation, research questions and objectives, viva preparation, and report Sections I & II (`introduction.tex` and `related_work.tex`).
- **Member 2:** Dataset acquisition, exploratory data analysis (EDA), data quality checks, and report Section III (`dataset_eda.tex`).
- **Member 3:** Preprocessing pipeline, feature engineering, missing-signal sentinel handling, ablation experiments, and report Section IV (`feature_engineering.tex`).
- **Team:** Methodology, modeling (KNN, Random Forest, XGBoost, Single-Task DNN, Multitask DNN), results, discussion, and conclusion.
