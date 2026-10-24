# Crime Hotspot Classification Report (Part 4)

**Project Title:** Comparative Analysis of Machine Learning Models for Crime Category Classification and Spatial Hotspot Classification Across Four Regions  
**Date:** October 2026  
**Status:** Part 4 Completed — Hotspot Classification, Leakage-Free Temporal Engineering, DBSCAN Analysis, and Regional Benchmarking  

---

## 1. Executive Summary

This report documents the second core machine learning task: **Crime Hotspot Classification**.

The objective is to classify geographic spatial units into **three operational crime concentration tiers**:
1. **Low Crime Concentration**
2. **Medium Crime Concentration**
3. **High Crime Concentration**

### Strict Implementation Rules Adhered to:
- **Independent Modeling:** Separate models and preprocessing pipelines were executed across Chicago, Los Angeles, San Francisco, and the UK.
- **Incident Concentration Target Definition:** Because fine-grained census population data is unavailable at the 1 km grid/beat level in public police dispatch feeds, the target represents **incident concentration** rather than population-adjusted per-capita risk.
- **Quantile-Based Discretization:** Documented tertile cutoffs (33.3% and 66.7% percentiles of recent crime volumes) were utilized, guaranteeing balanced, robust class definitions across regions.
- **Strict Leakage Prevention:** All historical predictors (`prev_7d_crime_count`, `prev_30d_crime_count`, `prev_7d_theft_count`, `prev_7d_assault_count`, `prev_period_crime_count`) were calculated strictly using past records via `pandas.rolling(window, closed='left')` on sorted datetime indices. No future information was used.
- **Exploratory Spatial Clustering:** DBSCAN clustering was performed on geographic coordinates using the spherical Haversine metric ($R = 6371\text{ km}$, $\epsilon = 1.0\text{ km}$). Clusters were evaluated as exploratory density groupings rather than assumed statistical significance.
- **Primary Metric:** **Macro-F1** across the 3 concentration tiers.

---

## 2. Geographic Units and Discretization

| Dataset | Primary Geographic Unit | Spatial Grid Size | Coordinate Availability | Quantile Tertiles (33.3% / 66.7%) |
| :--- | :--- | :--- | :--- | :--- |
| **Chicago** | Police District & 0.01° Grid | $\approx 1.1\text{ km}$ cells | 99.8% valid | 33rd: 16.0 crimes / 67th: 31.0 crimes |
| **Los Angeles** | LAPD Division & 0.01° Grid | $\approx 1.1\text{ km}$ cells | 99.5% valid | 33rd: 73.0 crimes / 67th: 96.0 crimes |
| **San Francisco** | SFPD Police District & 0.01° Grid | $\approx 1.1\text{ km}$ cells | 42.2% valid (online reports omit coords) | 33rd: 9.0 crimes / 67th: 19.0 crimes |
| **UK** | Home Office Street Unit & 0.02° Grid | $\approx 2.2\text{ km}$ cells | 100.0% valid | 33rd: 1.0 crimes / 67th: 7.0 crimes |

---

## 3. Comparative Performance Tables by Region

### 3.1 Chicago Hotspot Classification
- **Chronological Split:** 32,000 train records / 8,000 test records.
- **Quantile Thresholds:** Low ($\le 16$), Medium ($17 - 31$), High ($> 31$).

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 (Primary) | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.8886** | **0.8507** | **0.8298** | **0.8389** | **0.8869** |
| Random Forest | 0.8476 | 0.7968 | 0.8024 | 0.7990 | 0.8467 |
| Decision Tree | 0.8374 | 0.7761 | 0.7793 | 0.7776 | 0.8385 |
| Logistic Regression | 0.8214 | 0.7847 | 0.8330 | 0.7810 | 0.8208 |

*Chicago Takeaways:* XGBoost achieved an outstanding Macro-F1 of 0.8389 and Accuracy of 88.86%. The most critical features were `prev_30d_crime_count`, `prev_7d_theft_count`, and `prev_period_crime_count`.

---

### 3.2 Los Angeles Hotspot Classification
- **Chronological Split:** 32,000 train records / 8,000 test records.
- **Quantile Thresholds:** Low ($\le 73$), Medium ($74 - 96$), High ($> 96$).

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 (Primary) | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.8728** | **0.8251** | **0.8558** | **0.8367** | **0.8763** |
| Logistic Regression | 0.8403 | 0.7967 | 0.8376 | 0.8105 | 0.8464 |
| Random Forest | 0.8034 | 0.7554 | 0.8033 | 0.7660 | 0.8119 |
| Decision Tree | 0.8011 | 0.7464 | 0.7814 | 0.7567 | 0.8068 |

*Los Angeles Takeaways:* XGBoost led with Macro-F1: 0.8367 and Accuracy: 87.28%. Logistic Regression also performed strongly (Macro-F1: 0.8105), demonstrating strong linear correlation between rolling 30-day divisional crime counts and short-term tier risk.

---

### 3.3 San Francisco Hotspot Classification
- **Chronological Split:** 32,000 train records / 8,000 test records.
- **Quantile Thresholds:** Low ($\le 9$), Medium ($10 - 19$), High ($> 19$).

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 (Primary) | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.8631** | **0.8428** | **0.7714** | **0.8010** | **0.8601** |
| Random Forest | 0.8596 | 0.8355 | 0.7575 | 0.7890 | 0.8558 |
| Decision Tree | 0.8482 | 0.8170 | 0.7427 | 0.7727 | 0.8443 |
| Logistic Regression | 0.8502 | 0.7728 | 0.7301 | 0.7485 | 0.8467 |

*San Francisco Takeaways:* XGBoost achieved Macro-F1: 0.8010 and Accuracy: 86.31%. Police district historical theft and violent incident volume effectively stabilized hotspot predictions across periods.

---

### 3.4 UK Hotspot Classification
- **Chronological Split:** 32,000 train records / 8,000 test records.
- **Quantile Thresholds:** Low ($\le 1$), Medium ($2 - 7$), High ($> 7$).

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 (Primary) | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 0.9483 | 0.9418 | **0.9444** | **0.9431** | **0.9484** |
| **XGBoost** | **0.9484** | **0.9440** | 0.9413 | 0.9426 | 0.9483 |
| Decision Tree | 0.9346 | 0.9314 | 0.9220 | 0.9262 | 0.9342 |
| Logistic Regression | 0.8579 | 0.8930 | 0.8042 | 0.8137 | 0.8399 |

*UK Takeaways:* Tree ensembles demonstrated near-identical top-tier performance on the UK data (Random Forest Macro-F1: 0.9431, XGBoost Macro-F1: 0.9426, Accuracy: 94.84%). Street-anchor historical crime persistence provided strong predictive separation.

---

## 4. Exploratory Spatial Analysis with DBSCAN

DBSCAN was run on subsampled coordinate arrays using the Haversine metric with an $\epsilon = 1.0\text{ km}$ neighborhood threshold and `min_samples = 30`:

- **Chicago:** Identified **7 primary spatial clusters** and 302 noise points. High-density clusters concentrated in the Chicago Loop, Near North Side, and South Side corridors.
- **Los Angeles:** Identified **8 spatial clusters** and 1,276 noise points. Major density clusters aligned with Downtown LA / Central, Hollywood, Van Nuys, and South LA (77th Street).
- **San Francisco:** Identified **2 primary spatial clusters** and 6 noise points. Clusters concentrated in the northeastern downtown/Tenderloin/Mission dense grid.
- **UK:** Identified **14 discrete urban spatial clusters** and only 1 noise point. Distinct clusters directly mapped to individual metro areas (Greater London, Birmingham, Leeds, Liverpool, Bristol, Newcastle, Sheffield, Nottingham, Cardiff).

*Interpretation Note:* In crime science, DBSCAN groupings reflect spatial proximity of recorded events. They are exploratory indicators of geographic density rather than confirmed statistical anomalies (e.g., Getis-Ord $G_i^*$).

---

## 5. Summary Comparison: Task 1 vs. Task 2

```
========================================================================================
CROSS-TASK PERFORMANCE COMPARISON ACROSS FOUR REGIONS (Best Model: XGBoost / RF)
========================================================================================
Region            Task 1: Category Macro-F1    Task 2: Hotspot Macro-F1    Hotspot Acc
----------------------------------------------------------------------------------------
Chicago           0.2486                       0.8389                      88.86%
Los Angeles       0.4644                       0.8367                      87.28%
San Francisco     0.3308                       0.8010                      86.31%
UK                0.2408                       0.9431                      94.84%
========================================================================================
```

### Synthesis:
1. **Predictability Gap:** Crime hotspot concentration (Task 2) exhibits significantly higher predictive accuracy (86%–95%) and Macro-F1 (0.80–0.94) compared to crime category classification (Task 1). This aligns with spatial criminology principles: **where crime occurs is far more regular and temporally persistent than the specific category of the next incident**.
2. **Feature Importance Uniformity:** In all four regions, rolling historical counts (`prev_30d_crime_count`, `prev_7d_theft_count`, `prev_period_crime_count`) accounted for over 70% of feature importance in the gradient-boosted trees.
3. **Operational Utility:** The 3-tier classification offers direct utility for law enforcement resource deployment, allowing proactive patrol allocation to predicted "High Concentration" zones prior to upcoming operational shifts.

---

## 6. Generated Artifacts

- **Model Checkpoints:**
  - `models/hotspot_classification/<region>/` (16 models: `logistic_regression.joblib`, `decision_tree.joblib`, `random_forest.joblib`, `xgboost.joblib`)
- **Metric Summaries:**
  - `results/metrics/hotspot_classification/<region>/model_comparison.csv`
  - `results/metrics/hotspot_classification/<region>/<model>_feature_importance.csv`
  - `results/metrics/hotspot_classification/all_regions_summary.json`
- **Spatial Figures & Maps:**
  - `results/figures/hotspot_classification/<region>/spatial_hotspot_visualization.png` (Dual plot: DBSCAN clusters and 3-Tier Hotspot maps)
  - `results/figures/hotspot_classification/<region>/<model>_cm.png` (Confusion matrices)
  - `results/figures/hotspot_classification/<region>/<model>_feature_importance.png` (Top 10 features)
  - `results/figures/hotspot_classification/<region>/model_comparison_plot.png` (Performance comparison)
- **Source Code:**
  - [`src/hotspot_classification.py`](file:///c:/Users/darsh/OneDrive/Desktop/crime/src/hotspot_classification.py)
