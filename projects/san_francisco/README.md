# San Francisco Crime Classification & Hotspot Analysis

This folder contains the complete, standalone end-to-end crime analysis notebook and dataset for **San Francisco, USA**.

## 📁 Folder Contents
- `San_Francisco_Crime_Analysis.ipynb`: Complete self-contained Jupyter Notebook covering:
  1. **Dataset Acquisition & Inspection** (DataSF / JSON)
  2. **Data Preprocessing & Feature Engineering** (Zero-leakage temporal rolling features, spatial validation)
  3. **Task 1: Crime Category Classification** (Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost)
  4. **Task 2: Hotspot Classification** (3 Concentration Tiers: Low, Medium, High)
  5. **Task 3: Spatial Density Clustering** (DBSCAN with Haversine distance)
- `data/`:
  - `san_francisco_crimes_raw.json`: Raw downloaded crime records
  - `san_francisco_processed.csv`: Cleaned dataset
  - `san_francisco_task1_crime_category.csv`: Feature matrix for multiclass category classification
  - `san_francisco_task2_hotspot.csv`: Feature matrix for 3-tier hotspot classification

## 🚀 How to Run
1. Open terminal inside this folder:
   ```bash
   cd projects/san_francisco
   ```
2. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook San_Francisco_Crime_Analysis.ipynb
   ```
3. Run all cells (`Cell` -> `Run All`).
