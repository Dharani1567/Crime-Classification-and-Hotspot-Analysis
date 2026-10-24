# Unsupervised Crime Spatial Clustering Report Using DBSCAN

**Project Title:** Comparative Analysis of Machine Learning Models for Crime Category Classification and Spatial Hotspot Classification Across Four Regions  
**Date:** October 2026  
**Status:** Part 5 (Unsupervised Task) Completed — Spatial Density Clustering, Parameter Exploration, Category Profiling, and Cross-Jurisdiction Analysis  

---

## 1. Executive Summary & Architectural Scope

This report presents the third independent analytical component of the project: **Unsupervised Crime Spatial Clustering Using DBSCAN** (Density-Based Spatial Clustering of Applications with Noise).

### Architectural Isolation & Methodological Guarantees:
1. **Independent Spatial Task:** This unsupervised spatial density discovery component is kept completely separate from:
   - Supervised Task 1: Crime Category Classification
   - Supervised Task 2: Crime Hotspot Classification
2. **Regional Autonomy:** DBSCAN was executed independently on **Chicago**, **Los Angeles**, **San Francisco**, and the **UK (Greater London Metro)**. Regional boundaries were strictly preserved; no cross-border coordinates were merged.
3. **Haversine Geodesic Metric:** All geographic coordinates (latitude and longitude) were transformed into **radians** to compute true spherical geodesic distances on Earth ($R = 6,371.0088\text{ km}$), avoiding Euclidean planar projection distortions.
4. **Parameter Grid Exploration:** Multiple combinations of distance threshold ($\epsilon$) and density threshold (`min_samples`) were empirically evaluated for each region to determine the optimal density resolution.
5. **No Synthetic Data:** Genuine, validated geographic coordinates from Part 2 were utilized.
6. **Scientific Interpretation Safeguard:**
   - DBSCAN cluster IDs are **arbitrary nominal identifiers**, not severity or risk rankings.
   - DBSCAN clusters represent **spatial density concentrations**, not statistically verified hotspots (such as Getis-Ord $G_i^*$).
   - DBSCAN does not predict future events; it discovers geographic patterns within historical observations.

---

## 2. DBSCAN Parameter Sensitivity Analysis

Because urban density, geographic footprint, and coordinate distributions vary significantly across cities, a single fixed parameter set cannot be applied uniformly. A multi-configuration sensitivity sweep was performed for each region:

### 2.1 Parameter Sweep Results

| Dataset | Tested Configuration ($\epsilon$, `min_samples`) | Clusters Found | Noise Points | Noise (%) | Silhouette Score | Selected Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chicago** | $\epsilon = 0.3\text{ km}, \text{ms} = 25$ | 65 | 8,970 | 59.8% | 0.4158 | Over-fragmented |
| **Chicago** | **$\epsilon = 0.5\text{ km}, \text{ms} = 30$** | **31** | **3,471** | **23.1%** | **-0.1006** | **Selected Primary** |
| **Chicago** | $\epsilon = 0.8\text{ km}, \text{ms} = 50$ | 5 | 1,665 | 11.1% | -0.1841 | Over-merged |
| **Chicago** | $\epsilon = 1.0\text{ km}, \text{ms} = 50$ | 3 | 765 | 5.1% | 0.1952 | Macro-basin merge |
| **Los Angeles** | $\epsilon = 0.4\text{ km}, \text{ms} = 30$ | 51 | 8,460 | 56.4% | 0.4586 | High noise drop |
| **Los Angeles** | **$\epsilon = 0.6\text{ km}, \text{ms} = 40$** | **34** | **5,571** | **37.1%** | **0.4137** | **Selected Primary** |
| **Los Angeles** | $\epsilon = 0.8\text{ km}, \text{ms} = 50$ | 22 | 3,720 | 24.8% | 0.2492 | Basin consolidation |
| **Los Angeles** | $\epsilon = 1.2\text{ km}, \text{ms} = 60$ | 5 | 1,770 | 11.8% | 0.3498 | Over-merged |
| **San Francisco** | $\epsilon = 0.2\text{ km}, \text{ms} = 20$ | 48 | 2,985 | 19.9% | -0.1054 | Micro-block splits |
| **San Francisco** | **$\epsilon = 0.35\text{ km}, \text{ms} = 30$** | **15** | **1,036** | **6.9%** | **0.2968** | **Selected Primary** |
| **San Francisco** | $\epsilon = 0.5\text{ km}, \text{ms} = 40$ | 8 | 375 | 2.5% | 0.2915 | Peninsula merge |
| **San Francisco** | $\epsilon = 0.8\text{ km}, \text{ms} = 50$ | 2 | 15 | 0.1% | 0.4559 | Monolithic cluster |
| **UK (London)** | $\epsilon = 0.25\text{ km}, \text{ms} = 30$ | 9 | 270 | 1.8% | 0.3569 | Moderate splits |
| **UK (London)** | **$\epsilon = 0.4\text{ km}, \text{ms} = 40$** | **3** | **16** | **0.1%** | **0.6373** | **Selected Primary** |
| **UK (London)** | $\epsilon = 0.6\text{ km}, \text{ms} = 50$ | 3 | 0 | 0.0% | 0.6395 | Uniform coverage |
| **UK (London)** | $\epsilon = 0.8\text{ km}, \text{ms} = 60$ | 3 | 0 | 0.0% | 0.6358 | Uniform coverage |

---

## 3. Detailed Cluster Statistics & Spatial Profiles

### 3.1 City of Chicago
- **Selected Configuration:** $\epsilon = 0.5\text{ km}$ ($500\text{ m}$), `min_samples = 30`.
- **Clustering Summary:**
  - Evaluated Records: 15,000 valid coordinate points.
  - **Clusters Discovered:** **31 discrete spatial clusters**.
  - **Noise Points:** **3,471 incidents (23.14%)**.
  - Clustered Crime Volume: 11,529 incidents (76.86%).
- **Representative Top Clusters:**
  1. **Cluster 1 (North/Loop/West Corridors):** 5,232 incidents (45.38% of clustered volume). Median location: (41.87974° N, -87.68974° W). Est. Density: $31.9\text{ crimes/km}^2$. Top categories: Battery (1,154), Theft (1,102), Criminal Damage (587).
  2. **Cluster 0 (South/Southeast Corridors):** 3,283 incidents (28.48% of clustered volume). Median location: (41.76512° N, -87.61349° W). Est. Density: $24.4\text{ crimes/km}^2$. Top categories: Battery (784), Theft (593), Weapons Violation (381).
  3. **Cluster 3 (Northwest Cluster):** 270 incidents. Median location: (41.92110° N, -87.69104° W). Est. Density: $29.5\text{ crimes/km}^2$. Top categories: Theft (82), Criminal Damage (35).
- **Geographic Observations:** High spatial cohesion along historical arterial thoroughfares (Michigan Ave, Madison St, State St). Violent crime categories (Weapons Violations, Battery) exhibit intense localized clustering in the South Side, whereas Property Theft clusters tightly around commercial centers.

---

### 3.2 City of Los Angeles
- **Selected Configuration:** $\epsilon = 0.6\text{ km}$ ($600\text{ m}$), `min_samples = 40`.
- **Clustering Summary:**
  - Evaluated Records: 15,000 valid coordinate points.
  - **Clusters Discovered:** **34 discrete spatial clusters**.
  - **Noise Points:** **5,571 incidents (37.14%)**.
  - Clustered Crime Volume: 9,429 incidents (62.86%).
- **Representative Top Clusters:**
  1. **Cluster 1 (Central / Downtown LA / 77th St):** 3,008 incidents (31.90% of clustered volume). Median location: (34.02510° N, -118.28600° W). Est. Density: $28.0\text{ crimes/km}^2$. Top categories: Battery Simple Assault (482), Intimate Partner Assault (345), Robbery (291).
  2. **Cluster 3 (Hollywood / Wilshire Corridor):** 984 incidents (10.44% of clustered volume). Median location: (34.08920° N, -118.37740° W). Est. Density: $21.9\text{ crimes/km}^2$. Top categories: Burglary from Vehicle (185), Theft Plain Petty (124).
  3. **Cluster 4 (San Fernando Valley / Van Nuys):** 742 incidents (7.87% of clustered volume). Median location: (34.20140° N, -118.44850° W). Est. Density: $25.1\text{ crimes/km}^2$. Top categories: Vehicle Stolen (198), Burglary (112).
- **Geographic Observations:** LA's vast geographic surface area ($\approx 1,214\text{ km}^2$) results in higher noise dispersion (37.14%) across residential hills and sprawling suburbs. Clustered activity isolates distinct commercial centers (Hollywood Blvd, DTLA Skid Row, Venice Beach, Van Nuys).

---

### 3.3 City of San Francisco
- **Selected Configuration:** $\epsilon = 0.35\text{ km}$ ($350\text{ m}$), `min_samples = 30`.
- **Clustering Summary:**
  - Evaluated Records: 15,000 valid coordinate points.
  - **Clusters Discovered:** **15 discrete spatial clusters**.
  - **Noise Points:** **1,036 incidents (6.91%)**.
  - Clustered Crime Volume: 13,964 incidents (93.09%).
- **Representative Top Clusters:**
  1. **Cluster 0 (Downtown / Tenderloin / Mission Core):** 12,214 incidents (87.47% of clustered volume). Median location: (37.78120° N, -122.41726° W). Est. Density: **$125.9\text{ crimes/km}^2$**. Top categories: Larceny Theft (4,120), Assault (1,240), Drug Offense (980).
  2. **Cluster 1 (South Beach / Embarcadero Waterfront):** 910 incidents (6.52% of clustered volume). Median location: (37.78840° N, -122.39157° W). Est. Density: $63.9\text{ crimes/km}^2$. Top categories: Larceny Theft (512), Malicious Mischief (84).
  3. **Cluster 3 (Richmond District Commercial):** 94 incidents. Median location: (37.78200° N, -122.45716° W). Est. Density: $84.6\text{ crimes/km}^2$. Top categories: Larceny Theft (48), Burglary (18).
- **Geographic Observations:** San Francisco displays the highest spatial concentration in the study: **87.5% of all clustered incidents reside within a single hyper-dense continuous core (Cluster 0)** encompassing the Tenderloin, SOMA, and Mission corridors.

---

### 3.4 United Kingdom (Greater London Metropolitan Area)
- **Selected Configuration:** $\epsilon = 0.4\text{ km}$ ($400\text{ m}$), `min_samples = 40`.
- **Regional Isolation:** Restricted specifically to the Greater London Metropolitan zone (lat [51.35, 51.65], lon [-0.25, 0.05]) to prevent invalid clustering across disjoint nationwide cities.
- **Clustering Summary:**
  - Evaluated Records: 15,000 valid coordinate points.
  - **Clusters Discovered:** **3 primary spatial clusters**.
  - **Noise Points:** **16 incidents (0.11%)**.
  - Clustered Crime Volume: 14,984 incidents (99.89%).
- **Representative Clusters:**
  1. **Cluster 0 (Central London / Westminster / Camden Hub):** 11,587 incidents (77.33% of clustered volume). Median location: (51.51214° N, -0.12692° W). Est. Density: **$276.5\text{ crimes/km}^2$**. Top categories: Theft from the person (2,410), Violent crime (2,180), Other theft (1,920).
  2. **Cluster 2 (East London / Hackney / Tower Hamlets):** 1,965 incidents (13.11% of clustered volume). Median location: (51.54320° N, -0.05695° W). Est. Density: $202.4\text{ crimes/km}^2$. Top categories: Violent crime (640), Anti-social behaviour (410).
  3. **Cluster 1 (South London / Southwark / Lambeth):** 1,432 incidents (9.56% of clustered volume). Median location: (51.49810° N, -0.10106° W). Est. Density: $146.1\text{ crimes/km}^2$. Top categories: Violent crime (485), Public order (210).
- **Geographic Observations:** Because UK police reports snap coordinates to anonymized street centroids (e.g., "On or near Nightclub", "On or near Shopping Area"), multiple crimes coalesce around identical geographic reference anchors. Consequently, noise is minimal (0.11%), and clusters naturally outline major transit, retail, and nightlife hubs.

---

## 4. Cross-Regional Comparison Summary Table

| Metric / Dimension | Chicago | Los Angeles | San Francisco | UK (London Metro) |
| :--- | :--- | :--- | :--- | :--- |
| **Geographic Scope** | Municipal Chicago ($\approx 606\text{ km}^2$) | City of LA ($\approx 1,214\text{ km}^2$) | SF Peninsula ($\approx 121\text{ km}^2$) | Greater London Metro |
| **Sample Size Evaluated** | 15,000 | 15,000 | 15,000 | 15,000 |
| **Optimal Distance Threshold ($\epsilon$)** | $0.5\text{ km}$ ($500\text{ m}$) | $0.6\text{ km}$ ($600\text{ m}$) | $0.35\text{ km}$ ($350\text{ m}$) | $0.4\text{ km}$ ($400\text{ m}$) |
| **Minimum Points (`min_samples`)** | 30 | 40 | 30 | 40 |
| **Clusters Identified** | **31** | **34** | **15** | **3** |
| **Noise Points Count** | 3,471 | 5,571 | 1,036 | 16 |
| **Noise Percentage (%)** | 23.14% | 37.14% | 6.91% | 0.11% |
| **Clustered Crimes Count** | 11,529 (76.86%) | 9,429 (62.86%) | 13,964 (93.09%) | 14,984 (99.89%) |
| **Largest Cluster Share** | 45.38% (Cluster 1) | 31.90% (Cluster 1) | 87.47% (Cluster 0) | 77.33% (Cluster 0) |
| **Peak Estimated Density** | $37.4\text{ crimes/km}^2$ | $37.4\text{ crimes/km}^2$ | $125.9\text{ crimes/km}^2$ | $276.5\text{ crimes/km}^2$ |
| **Silhouette Score** | -0.1006 | +0.4137 | +0.2968 | +0.6373 |

---

## 5. Architectural Integration of Three Analytical Components

The completed project codebase now integrates three complementary components:

```
========================================================================================================
FINAL MACHINE LEARNING SYSTEM ARCHITECTURE (Three Isolated Analytical Components)
========================================================================================================

Component 1: Supervised Crime Category Classification
   - Algorithm: XGBoost / Random Forest (Champion: XGBoost, Macro-F1 up to 0.4644 in LA)
   - Inputs: Temporal cycles, location frequency encodings, victim demographics, premise/weapon tags
   - Target: Clean multi-class crime offense category

Component 2: Supervised Crime Hotspot Classification
   - Algorithm: XGBoost / Random Forest (Champion: XGBoost, Accuracy up to 94.84% in UK, 88.86% in Chicago)
   - Inputs: Leakage-free past rolling aggregations (prev 7-day, 30-day, theft, assault counts)
   - Target: 3-tier ordinal concentration (Low, Medium, High Concentration)

Component 3: Unsupervised Spatial Clustering (DBSCAN)
   - Algorithm: Geodesic Haversine DBSCAN
   - Inputs: Spherical coordinate arrays (radians)
   - Output: Arbitrary nominal spatial density cluster assignments, noise filtering, and density extents
========================================================================================================
```

---

## 6. Generated Output Files

All artifacts for Part 5 (Unsupervised Task) are saved:

- **Source Code:**
  - [`src/dbscan_clustering.py`](file:///c:/Users/darsh/OneDrive/Desktop/crime/src/dbscan_clustering.py)
- **Cluster Labels & Statistics:**
  - `results/dbscan/<region>/cluster_labels.csv`
  - `results/dbscan/<region>/cluster_statistics.csv`
  - `results/dbscan/<region>/parameter_comparison.csv`
  - `results/dbscan/all_dbscan_summary.json`
- **Visual Maps & Diagnostic Figures:**
  - `results/dbscan/<region>/dbscan_cluster_map.png` (Incident map with cluster colors and grey noise points)
  - `results/dbscan/<region>/cluster_crime_category_composition.png` (Offense distributions per top cluster)
  - `results/dbscan/<region>/parameter_sensitivity.png` (Bar and line chart showing clusters vs noise %)
- **Technical Report:**
  - [`results/reports/dbscan_analysis.md`](file:///c:/Users/darsh/OneDrive/Desktop/crime/results/reports/dbscan_analysis.md)
