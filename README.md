# Comparative Analysis of Machine Learning Models for Crime Category Classification and Spatial Hotspot Classification Across Four Regions

A production-grade machine learning project implementing independent pipelines, strict leakage-free temporal feature engineering, and rigorous model benchmarking across four international metropolitan jurisdictions:
1. **Chicago, USA** (City of Chicago Data Portal)
2. **Los Angeles, USA** (City of Los Angeles Open Data)
3. **San Francisco, USA** (DataSF)
4. **United Kingdom** (data.police.uk / Home Office)

---

## 📌 Project Overview & Architectural Principles

- **No Dataset Merging:** Each dataset is maintained and modeled completely independently, retaining its native reporting schemas, statutory penal codes, and geographic reference systems.
- **Two Distinct Machine Learning Tasks:**
  1. **Task 1: Crime Category Classification (Multiclass):** Predicts the specific crime category using contextual, temporal, and spatial descriptors.
  2. **Task 2: Crime Hotspot Classification (3 Concentration Tiers):** Classifies geographic units into Low, Medium, and High incident concentration.
- **Strict Leakage Prevention:**
  - Forward-looking chronological train/test splitting (80% train / 20% test).
  - All rolling historical aggregations (`prev_7d`, `prev_30d`) are computed using `closed='left'` on sorted datetime indices, strictly referencing past events.
  - Frequency encoders, target transformers, and feature scalers are fitted exclusively on training sets.
- **Primary Metric:** **Macro-F1** is prioritized over raw accuracy to prevent majority-class bias on imbalanced crime distributions.

---

## 📊 Summary of Benchmark Results

### Task 1: Crime Category Classification (Multiclass)
| Dataset | Best Architecture | Accuracy | Macro-F1 (Primary) | Weighted-F1 | Macro Precision | Macro Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chicago** | **XGBoost** | **38.49%** | **0.2486** | 0.3403 | 0.3139 | 0.2538 |
| **Los Angeles** | **XGBoost** | **57.17%** | **0.4644** | 0.5316 | 0.4933 | 0.4878 |
| **San Francisco** | **XGBoost** | **61.68%** | **0.3308** | 0.5678 | 0.4221 | 0.3286 |
| **UK** | **XGBoost** | **44.64%** | **0.2408** | 0.3942 | 0.2978 | 0.2560 |

### Task 2: Crime Hotspot Classification (3 Tiers: Low, Medium, High)
| Dataset | Best Architecture | Accuracy | Macro-F1 (Primary) | Weighted-F1 | Macro Precision | Macro Recall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chicago** | **XGBoost** | **88.86%** | **0.8389** | 0.8869 | 0.8507 | 0.8298 |
| **Los Angeles** | **XGBoost** | **87.28%** | **0.8367** | 0.8763 | 0.8251 | 0.8558 |
| **San Francisco** | **XGBoost** | **86.31%** | **0.8010** | 0.8601 | 0.8428 | 0.7714 |
| **UK** | **Random Forest / XGBoost** | **94.84%** | **0.9431** | 0.9484 | 0.9418 | 0.9444 |

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                           # Raw unadulterated JSON downloads
│   ├── processed/                     # Cleaned, validated tabular CSVs per region
│   └── features/                      # Task 1 & Task 2 engineered feature matrices
├── models/
│   ├── crime_classification/          # 20 trained models across 4 regions
│   └── hotspot_classification/        # 16 trained models across 4 regions
├── results/
│   ├── figures/                       # Confusion matrices, ROC/PR, DBSCAN & Hotspot maps
│   │   ├── crime_classification/
│   │   ├── hotspot_classification/
│   │   └── model_comparison/
│   ├── metrics/                       # Detailed classification reports and CSVs
│   │   ├── crime_classification/
│   │   ├── hotspot_classification/
│   │   └── model_comparison.csv
│   └── reports/                       # Technical reports for Parts 1 to 5
│       ├── dataset_summary.csv
│       ├── dataset_inspection.md
│       ├── preprocessing_report.md
│       ├── crime_classification_report.md
│       ├── hotspot_classification_report.md
│       └── comparative_analysis.md
├── scripts/
│   ├── download_datasets.py           # Programmatic API downloader (Socrata & UK Police)
│   └── inspect_datasets.py            # Integrity and schema verification
└── src/
    ├── preprocessing.py               # Independent preprocessing pipelines
    ├── feature_engineering.py          # Leakage-free spatiotemporal feature engineering
    ├── crime_classification.py         # Multi-model training for Task 1
    ├── hotspot_classification.py       # Multi-model training & DBSCAN for Task 2
    └── comparative_evaluation.py       # Cross-regional benchmark consolidation
```

---

## 🚀 Getting Started

### 1. Requirements
Ensure Python 3.10+ is installed with the required libraries:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn requests joblib
```

### 2. Download Raw Datasets
```bash
python scripts/download_datasets.py
```

### 3. Preprocess & Feature Engineering
```bash
python src/preprocessing.py
python src/feature_engineering.py
```

### 4. Train Models
```bash
python src/crime_classification.py
python src/hotspot_classification.py
```

### 5. Generate Comparative Reports & Figures
```bash
python src/comparative_evaluation.py
```

---

## 📜 Technical Reports
Detailed markdown documentation is located in `results/reports/`:
- **Part 1:** [`dataset_inspection.md`](results/reports/dataset_inspection.md)
- **Part 2:** [`preprocessing_report.md`](results/reports/preprocessing_report.md)
- **Part 3:** [`crime_classification_report.md`](results/reports/crime_classification_report.md)
- **Part 4:** [`hotspot_classification_report.md`](results/reports/hotspot_classification_report.md)
- **Part 5:** [`comparative_analysis.md`](results/reports/comparative_analysis.md)
