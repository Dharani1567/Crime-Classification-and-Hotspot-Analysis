# Crime Category Classification Report (Part 3)

**Project Title:** Comparative Analysis of Machine Learning Models for Crime Category Classification and Spatial Hotspot Classification Across Four Regions  
**Date:** October 2026  
**Status:** Part 3 Completed — Model Training, Chronological Evaluation, and Regional Comparative Benchmarking  

---

## 1. Executive Summary

This report presents the implementation and comparative evaluation of the first primary machine learning task: **Crime Category Classification**.

Five distinct model families were trained independently on each region:
1. **Logistic Regression** (Regularized multi-class linear baseline)
2. **Decision Tree** (Non-linear interpretable rule partitioning)
3. **Random Forest** (Bootstrap aggregation of deep decision trees)
4. **Support Vector Machine** (Linear Support Vector Classifier with probability calibration)
5. **XGBoost** (Extreme Gradient Boosted trees)

### Strict Architectural Guidelines Followed:
- **Zero Merging:** Datasets were trained and evaluated completely independently across Chicago, Los Angeles, San Francisco, and the UK.
- **Original Categorization:** Used genuine categories from the official datasets (no synthesized labels).
- **Chronological Train/Test Split:** Standard 80% train / 20% test partition ordered temporally to evaluate forward-looking predictive ability.
- **Leakage-Free Feature Processing:** Numeric scalers and categorical frequency encoders were fit **strictly on the training partition** and applied to test sets.
- **Primary Metric:** **Macro-F1** was designated as the primary criterion to capture performance across both dominant and long-tail crime classes without class-frequency bias.

---

## 2. Comparative Performance Tables by Region

### 2.1 Chicago Crime Dataset
- **Evaluation Partition:** 32,000 train records / 8,000 test records.
- **Target Offenses:** Top 15 statutory Illinois UCR categories (Theft, Battery, Criminal Damage, Assault, Motor Vehicle Theft, Other Offense, Deceptive Practice, Robbery, Weapons Violation, Burglary, Narcotics, etc.).

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 (Primary) | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.3849** | **0.3139** | **0.2538** | **0.2486** | **0.3403** |
| Random Forest | 0.3724 | 0.3029 | 0.2235 | 0.2132 | 0.3108 |
| Decision Tree | 0.3453 | 0.2660 | 0.2102 | 0.2119 | 0.3144 |
| Logistic Regression | 0.3439 | 0.1769 | 0.1886 | 0.1599 | 0.2592 |
| Support Vector Machine | 0.3359 | 0.1939 | 0.1982 | 0.1587 | 0.2694 |

*Chicago Takeaways:* XGBoost achieved the highest Macro-F1 (0.2486) and Accuracy (38.49%). Location descriptors (`location_description_freq`), domestic violence flags (`domestic`), arrest history, and hour of the day were the most predictive features.

---

### 2.2 Los Angeles Crime Dataset
- **Evaluation Partition:** 32,000 train records / 8,000 test records.
- **Target Offenses:** Top 15 LAPD crime categories (Stolen Vehicle, Battery Simple Assault, Burglary, Burglary From Vehicle, Felony Vandalism, Identity Theft, Intimate Partner Assault, Aggravated Assault w/ Deadly Weapon, Robbery, Petty Theft, etc.).

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 (Primary) | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.5717** | **0.4933** | **0.4878** | **0.4644** | **0.5316** |
| Decision Tree | 0.5281 | 0.4432 | 0.4513 | 0.4317 | 0.4979 |
| Random Forest | 0.5447 | 0.4944 | 0.4530 | 0.4220 | 0.4945 |
| Logistic Regression | 0.4328 | 0.2865 | 0.3109 | 0.2673 | 0.3695 |
| Support Vector Machine | 0.3322 | 0.2590 | 0.2155 | 0.1612 | 0.2429 |

*Los Angeles Takeaways:* LA yielded the highest overall predictive performance across all four regions (XGBoost Macro-F1: 0.4644, Accuracy: 57.17%). The availability of premise descriptors (`premis_desc_freq`), weapon types (`weapon_desc_freq`), and victim demographics (`vict_age`, `vict_sex`) substantially boosted model distinguishability.

---

### 2.3 San Francisco Crime Dataset
- **Evaluation Partition:** 32,000 train records / 8,000 test records.
- **Target Offenses:** Top 15 SFPD incident categories (Larceny Theft, Recovered Vehicle, Lost Property, Malicious Mischief, Assault, Motor Vehicle Theft, Burglary, Fraud, Non-Criminal, Robbery, Warrant, Drug Offense, etc.).

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 (Primary) | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.6168** | **0.4221** | **0.3286** | **0.3308** | **0.5678** |
| Random Forest | 0.6156 | 0.3485 | 0.3131 | 0.3154 | 0.5581 |
| Decision Tree | 0.5999 | 0.3415 | 0.3005 | 0.3048 | 0.5502 |
| Support Vector Machine | 0.5866 | 0.2523 | 0.2486 | 0.2213 | 0.4862 |
| Logistic Regression | 0.5829 | 0.2098 | 0.2370 | 0.2104 | 0.4789 |

*San Francisco Takeaways:* High overall accuracy (61.68%) was driven by the strong dominance of Larceny Theft and Recovered Vehicle cases. XGBoost achieved a Macro-F1 of 0.3308. Primary drivers included police district (`police_district_freq`), report type, online filing flags (`filed_online`), and diurnal temporal cycles.

---

### 2.4 UK Crime Dataset
- **Evaluation Partition:** 32,000 train records / 8,000 test records.
- **Target Offenses:** All 14 standardized UK Home Office categories (Violent Crime, Anti-Social Behaviour, Theft from the Person, Other Theft, Shoplifting, Public Order, Vehicle Crime, Drugs, Robbery, Burglary, Criminal Damage/Arson, Other Crime, Possession of Weapons, Bicycle Theft).

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 (Primary) | Weighted-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.4464** | **0.2978** | **0.2560** | **0.2408** | **0.3942** |
| Decision Tree | 0.4270 | 0.2687 | 0.2588 | 0.2388 | 0.3935 |
| Random Forest | 0.4293 | 0.2210 | 0.2247 | 0.2080 | 0.3784 |
| Logistic Regression | 0.3251 | 0.1060 | 0.1581 | 0.1063 | 0.2423 |
| Support Vector Machine | 0.1673 | 0.0901 | 0.1150 | 0.0613 | 0.1399 |

*UK Takeaways:* XGBoost and Decision Tree led performance (Macro-F1: 0.2408 and 0.2388). In the absence of hour-of-day timestamps, predictive power was concentrated in spatial coordinates (`latitude`, `longitude`), street POI types (`street_name_freq`), and outcome resolutions.

---

## 3. Cross-Regional Comparative Analysis

```
========================================================================================
MODEL BENCHMARK SUMMARY ACROSS FOUR INDEPENDENT DATASETS (Task 1: Crime Classification)
========================================================================================
Region            Best Model      Macro-F1 (Primary)    Accuracy    Weighted-F1
----------------------------------------------------------------------------------------
Chicago           XGBoost         0.2486                38.49%      0.3403
Los Angeles       XGBoost         0.4644                57.17%      0.5316
San Francisco     XGBoost         0.3308                61.68%      0.5678
UK                XGBoost         0.2408                44.64%      0.3942
========================================================================================
```

### Key Insights:
1. **Model Family Hierarchy:** XGBoost consistently proved to be the top-performing architecture across all four independent datasets, followed closely by Random Forest and Decision Trees. Tree-based ensembles handled non-linear spatial boundaries, frequency-encoded categorical distributions, and cyclical features significantly better than linear baselines (Logistic Regression and Linear SVM).
2. **Contextual Enrichment Multiplier:** Los Angeles models benefited heavily from rich situational context (victim age/sex and premise/weapon descriptions), yielding a ~20% higher Macro-F1 compared to models relying solely on spatiotemporal coordinates.
3. **Macro-F1 vs. Accuracy Dynamics:** In San Francisco, high overall Accuracy (61.68%) contrasted with a moderate Macro-F1 (0.3308) due to extreme class imbalance (Larceny Theft represented over 39% of cases). This underscores the critical importance of utilizing Macro-F1 as the primary metric.

---

## 4. Generated Artifacts & Directory Structure

All model checkpoints, metrics, and visual figures are stored in the following locations:

- **Trained Model Checkpoints:**
  - `models/crime_classification/chicago/` (5 models: `logistic_regression.joblib`, `decision_tree.joblib`, `random_forest.joblib`, `support_vector_machine.joblib`, `xgboost.joblib`)
  - `models/crime_classification/los_angeles/` (5 models)
  - `models/crime_classification/san_francisco/` (5 models)
  - `models/crime_classification/uk/` (5 models)
- **Evaluation Metrics & Reports:**
  - `results/metrics/crime_classification/<region>/model_comparison.csv`
  - `results/metrics/crime_classification/<region>/<model>_report.json`
  - `results/metrics/crime_classification/<region>/<model>_feature_importance.csv`
  - `results/metrics/crime_classification/all_regions_summary.json`
- **Visual Figures & Plots:**
  - `results/figures/crime_classification/<region>/model_comparison_plot.png`
  - `results/figures/crime_classification/<region>/<model>_cm.png` (Confusion Matrix plots)
  - `results/figures/crime_classification/<region>/<model>_feature_importance.png` (Top 12 Features)
- **Source Pipeline Script:**
  - [`src/crime_classification.py`](file:///c:/Users/darsh/OneDrive/Desktop/crime/src/crime_classification.py)
