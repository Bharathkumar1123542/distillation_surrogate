# Surrogate Model — Binary Distillation Column

**FOSSEE Screening Task | IIT Bombay**

A single-notebook, end-to-end ML surrogate that replaces expensive iterative
process simulations for a binary distillation column with fast ML inference.

This project is a screening assignment for the FOSSEE (Free and Open-Source Software for Science and Engineering Education) internship program at IIT Bombay.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Kaggle API credentials _(one-time)_

| OS            | Location                                  |
| ------------- | ----------------------------------------- |
| Windows       | `C:\Users\<username>\.kaggle\kaggle.json` |
| Linux / macOS | `~/.kaggle/kaggle.json`                   |

Steps:

1. Log in to [kaggle.com](https://www.kaggle.com) → **Account → API → Create New Token**
2. Save the downloaded `kaggle.json` to the path above.
3. Linux/macOS only: `chmod 600 ~/.kaggle/kaggle.json`
4. Verify: `kaggle datasets list` returns results without error.

> **No Kaggle access?** Manually download the CSV from  
> https://www.kaggle.com/datasets/amarhaiqal/aspen-hysys-distillation-column-data  
> and place it at `data/raw/distillation.csv`. Cell 1.2 will skip the API call automatically.

### 3. Run the notebook

```bash
jupyter notebook distillation_surrogate.ipynb
```

Use **Kernel → Restart & Run All** for a clean end-to-end execution.

---

## Project Structure

```
distillation-surrogate/
├── distillation_surrogate.ipynb   # Single deliverable — all code
├── requirements.txt               # Pinned dependencies
├── README.md                      # This file
│
├── data/
│   └── raw/
│       └── distillation.csv       # Downloaded by Cell Group 1
│
├── models/
│   ├── best_model.pkl             # joblib-serialized best surrogate
│   └── metrics_summary.json       # Test-set metrics for all models
│
└── figures/                       # Auto-generated plots
    ├── correlation_heatmap.png
    ├── distributions.png
    ├── missing_values_heatmap.png
    ├── parity_linreg.png
    ├── parity_polyreg.png
    ├── parity_rf.png
    ├── parity_xgb.png
    ├── parity_ann.png
    └── feature_importance.png
```

---

## Pipeline Stages

| Cell Group | Stage                               | Key Output                                    |
| ---------- | ----------------------------------- | --------------------------------------------- |
| 1          | Setup & Configuration               | Dependencies, constants, directories          |
| 2          | Dataset Acquisition                 | `data/raw/distillation.csv`                   |
| 3          | EDA & Data Validation               | Heatmaps, distributions, `COLUMN_MAP`         |
| 4          | Preprocessing & Feature Engineering | Scaled train/val/test splits                  |
| 5          | Model Training                      | 5 multi-output regressors                     |
| 6          | Evaluation & Physical Checks        | Metrics table, parity plots, 8 physical rules |
| 7          | Model Selection & Conclusion        | `best_model.pkl`, `metrics_summary.json`      |

---

## Models Trained

| ID        | Algorithm                           | Library      |
| --------- | ----------------------------------- | ------------ |
| `linreg`  | Linear Regression                   | scikit-learn |
| `polyreg` | Polynomial Regression (degree 2)    | scikit-learn |
| `rf`      | Random Forest (RandomizedSearchCV)  | scikit-learn |
| `xgb`     | XGBoost (RandomizedSearchCV)        | xgboost      |
| `ann`     | MLP Neural Network (early stopping) | scikit-learn |

---

## Inputs / Outputs

**Features (up to 10 with engineered feature):**
`T_feed`, `P_feed`, `z_F`, `N_stages`, `N_feed`, `RR`, `B_rate`, `q_feed`, `P_col`, `feed_stage_ratio`

**Targets:**
`x_D` (distillate purity), `x_B` (bottoms purity), `Q_C` (condenser duty), `Q_R` (reboiler duty)

## Dataset Citation

**Primary:** Amarhaiqal (2023). _Aspen HYSYS Distillation Column Data_.  
Kaggle. https://www.kaggle.com/datasets/amarhaiqal/aspen-hysys-distillation-column-data

**Alternate:** Jorgecote (2022). _Distillation Column_.  
Kaggle. https://www.kaggle.com/datasets/jorgecote/distillation-column
