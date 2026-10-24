# Data Preprocessing and Feature Engineering Report

**Project Title:** Comparative Analysis of Machine Learning Models for Crime Category Classification and Spatial Hotspot Classification Across Four Regions  
**Date:** October 2026  
**Status:** Part 2 Completed — Cleaned Datasets and Engineered Feature Sets Produced Independently  

---

## 1. Executive Summary

Part 2 completes the independent preprocessing and feature engineering for all four target datasets: **Chicago**, **Los Angeles**, **San Francisco**, and the **UK**.

Strict architecture rules have been adhered to:
1. **Four Independent Pipelines:** Each dataset was processed using dedicated logic without merging.
2. **Dedicated Target Datasets:** For every region, two distinct ML task feature sets were generated:
   - **Task 1: Crime Category Classification** (Multiclass classification predicting the clean crime category without proxy identifiers).
   - **Task 2: Crime Hotspot Classification** (Binary classification predicting high-density crime activity using past temporal and spatial indicators).
3. **Strict Leakage Prevention:**
   - Historical crime aggregations (previous 7-day, 30-day, and 60-day counts) were computed strictly using **past records** via rolling windows on sorted datetime indices with `closed='left'` (excluding the current event).
   - No target variable was included in the feature inputs.
   - Categorical transformations, cyclical encodings, and scaling are structured to fit exclusively on training sets.

---

## 2. Directory Layout of Generated Artifacts

```
crime/
├── src/
│   ├── preprocessing.py               (Independent cleaning & standardization scripts)
│   └── feature_engineering.py          (Leakage-free feature calculation for Task 1 & 2)
├── data/
│   ├── raw/                           (Unmodified original JSON files from Part 1)
│   ├── processed/
│   │   ├── chicago/
│   │   │   └── chicago_processed.csv               (100,000 records, 19 columns, 12.1 MB)
│   │   ├── los_angeles/
│   │   │   └── los_angeles_processed.csv           (100,000 records, 21 columns, 18.2 MB)
│   │   ├── san_francisco/
│   │   │   └── san_francisco_processed.csv         (98,712 records, 18 columns, 14.0 MB)
│   │   └── uk/
│   │       └── uk_processed.csv                    (174,216 records, 12 columns, 23.5 MB)
│   └── features/
│       ├── chicago/
│       │   ├── chicago_task1_crime_category.csv    (100,000 rows, 20 features + target, 18.5 MB)
│       │   └── chicago_task2_hotspot.csv           (100,000 rows, 12 features + target, 10.1 MB)
│       ├── los_angeles/
│       │   ├── los_angeles_task1_crime_category.csv(100,000 rows, 22 features + target, 24.8 MB)
│       │   └── los_angeles_task2_hotspot.csv       (100,000 rows, 12 features + target, 10.6 MB)
│       ├── san_francisco/
│       │   ├── san_francisco_task1_crime_category.csv (98,712 rows, 19 features + target, 20.3 MB)
│       │   └── san_francisco_task2_hotspot.csv     (98,712 rows, 14 features + target, 10.2 MB)
│       └── uk/
│           ├── uk_task1_crime_category.csv         (174,216 rows, 9 features + target, 24.7 MB)
│           └── uk_task2_hotspot.csv                (174,216 rows, 8 features + target, 14.3 MB)
└── results/
    └── reports/
        ├── dataset_summary.csv
        ├── dataset_inspection.md
        └── preprocessing_report.md
```

---

## 3. Preprocessing Operations & Duplicate / Null Treatment

### 3.1 Duplicate Removal
- **Chicago:** 0 duplicate primary IDs in raw extract.
- **Los Angeles:** 0 duplicate Division of Records numbers (`dr_no`).
- **San Francisco:** 1,177 duplicate row records removed (caused by supplementary or multi-charge amendments sharing identical `row_id`). Cleaned set: 98,712 records.
- **UK:** 0 duplicate primary crime incident IDs.

### 3.2 Missing Value Strategy
- **Chicago:**
  - `location_description`: Imputed with `OTHER/UNKNOWN`.
  - `district`, `ward`, `community_area`, `beat`: Imputed with sentinel identifiers (`UNKNOWN`, `-1`).
  - `latitude`, `longitude`: 228 records with null/zero coordinates were flagged via boolean `has_valid_coordinates=0` and bound-checked against lat [41.60, 42.10], lon [-88.00, -87.45].
- **Los Angeles:**
  - `premis_desc`: Imputed with `UNKNOWN/OTHER`.
  - `weapon_desc`: Imputed with `NO WEAPON REPORTED`.
  - `vict_age`: Imputed with `-1` (unreported).
  - `vict_sex`, `vict_descent`: Imputed with `UNKNOWN`.
  - Coordinates bound-checked against lat [33.65, 34.40], lon [-118.75, -118.10]. 451 non-physical `(0, 0)` points flagged via `has_valid_coordinates=0`.
- **San Francisco:**
  - `police_district`, `analysis_neighborhood`: Imputed with `UNKNOWN`.
  - `filed_online`: Imputed with `0` (False).
  - `latitude`, `longitude`: 57,007 online/unmapped reports flagged via `has_valid_coordinates=0` and bound-checked against lat [37.65, 37.85], lon [-122.55, -122.30]. Valid spatial coordinates retained: 41,705 (42.2%).
- **UK:**
  - Flattened nested JSON structure (`location.latitude`, `location.longitude`, `location.street.name`, `outcome_status.category`).
  - `street_name`: Imputed with `UNKNOWN`.
  - `outcome_category`: Imputed with `NO_OUTCOME_RECORDED`.
  - Coordinates verified within UK geographic bounding box (100% valid).

---

## 4. Feature Engineering & Leakage Prevention

### 4.1 Temporal Feature Extraction
Where timestamps exist (Chicago, LA, SF):
- Linear features: `hour`, `day_of_week_num` (0–6), `month`, `is_weekend` (binary 0/1).
- Continuous Cyclical Transformations:
  $$\text{sin\_hour} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{cos\_hour} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
  $$\text{sin\_month} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \quad \text{cos\_month} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$$
  $$\text{sin\_dayofweek} = \sin\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right), \quad \text{cos\_dayofweek} = \cos\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right)$$
- For the UK, month-level resolution is used (`month_num`, `sin_month`, `cos_month`).

### 4.2 Spatial Discretization
To model spatial distribution without overfitting to individual points:
- Continuous coordinates were discretized into uniform geographic bounding grids ($\Delta = 0.01^\circ \approx 1.1\text{ km}$ for cities, $\Delta = 0.02^\circ \approx 2.2\text{ km}$ for UK).
- Formed compound spatial cell IDs (e.g., `district_gridlat_gridlon`) reflecting local patrol beats and grid tiles.

### 4.3 Leakage-Free Historical Crime Calculations
To model hotspots:
- Datasets were sorted chronologically by datetime.
- Rolling crime aggregations (`past_crime_count_7d`, `past_crime_count_30d`, and `past_crime_count_60d`) were calculated per administrative division using `pandas.rolling(window, closed='left')`.
- Setting `closed='left'` strictly ensures that only records that occurred **before** the timestamp of the current record are included, eliminating lookahead bias.
- Hotspot classification targets were generated based on median past historical crime activity thresholds (yielding a balanced ~50/50 binary target distribution).

---

## 5. Machine Learning Feature Matrix Summary

| Dataset | Task 1: Category Classes | Task 1 Features | Task 2: Hotspot Target Rate | Task 2 Features |
| :--- | :--- | :--- | :--- | :--- |
| **Chicago** | 26 classes (rare <50 grouped to `OTHER_RARE`) | 20 features | 49.8% positive | 12 features |
| **Los Angeles** | 57 classes (rare <100 grouped to `OTHER_RARE`) | 22 features | 49.8% positive | 12 features |
| **San Francisco** | 35 classes (rare <50 grouped to `OTHER_RARE`) | 19 features | 48.5% positive | 14 features |
| **UK** | 14 standard Home Office classes | 9 features | 49.8% positive | 8 features |

---

## 6. Dataset Limitations & Modeling Recommendations

1. **Temporal Asynchrony:**
   - Chicago: Summer/Fall 2026.
   - LA: Full year 2020.
   - SF: Multi-year longitudinal (2018–2026).
   - UK: First half of 2024 (January–June).
   *Impact:* Cross-city comparisons evaluate how well models generalize within each region's specific operational context rather than matching a single calendar event.

2. **San Francisco Coordinate Sparsity:**
   - 57.8% of SF records lack geographic coordinates due to online self-reporting.
   *Impact:* For Task 1, `has_valid_coordinates` allows models to leverage district/neighborhood proxies. For Task 2, filtering to records with `has_valid_coordinates == 1` is recommended for high-resolution spatial hotspotting.

3. **Granular Label Long-Tail:**
   - LA contains 126 crime sub-codes in raw data. Grouping into the top 57 classes handles long tails while preserving granular offense types (e.g., separating vehicle theft from burglaries from motor vehicles).

4. **UK Temporal Granularity:**
   - The UK Police API records incidents at monthly intervals for privacy protection. Fine-grained diurnal (hourly) hotspotting is not supported; spatial hotspotting relies on multi-month rolling density and street-level points.

---

## 7. Compliance Checklist

- [x] Raw files loaded without altering original raw JSON data.
- [x] Exact duplicates removed (1,177 SF duplicate rows dropped; 0 in Chicago, LA, UK).
- [x] Systematic missing value imputation per domain feature.
- [x] Column names standardized across all four pipelines.
- [x] Datetime parsed and cyclical features extracted (`sin`/`cos` hour, day of week, month).
- [x] Spatial coordinates validated against regional bounding boxes.
- [x] Two distinct ML task datasets generated per region (saved in `data/features/<region>/`).
- [x] Historical rolling features calculated strictly on past records with `closed='left'` (zero leakage).
- [x] Target variables completely separated from input features.
- [x] **No ML models trained** (Stage strictly bounded to Part 2).
