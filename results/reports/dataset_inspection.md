# Dataset Inspection Report: Comparative Crime Analysis Across Four Regions

**Project Title:** Comparative Analysis of Machine Learning Models for Crime Category Classification and Spatial Hotspot Classification Across Four Regions  
**Date:** October 2026  
**Status:** Part 1 Completed — Acquisition, Integrity Verification, and Structural Inspection  

---

## 1. Executive Overview & Architecture Compliance

In accordance with strict project guidelines, all four regional crime datasets were acquired programmatically and are maintained **completely independently** in dedicated storage directories. No datasets are merged, and each dataset preserves its native recording ontology, schema, and geographic reference coordinates.

```
data/
└── raw/
    ├── chicago/
    │   └── chicago_crimes_raw.json           (100,000 records, 121.6 MB)
    ├── los_angeles/
    │   └── los_angeles_crimes_raw.json       (100,000 records, 71.2 MB)
    ├── san_francisco/
    │   └── san_francisco_crimes_raw.json     (100,000 records, 105.1 MB)
    └── uk/
        └── uk_crimes_raw.json                (174,216 records, 95.2 MB)
```

---

## 2. Dataset Summary Table

| Dataset | Records | Columns | Date Range | Crime Categories | Latitude Available | Longitude Available | Missing Values |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Chicago** | 100,000 | 31 | 2026-05-18 to 2026-10-15 | 31 | Yes (99,772 valid, 99.8%) | Yes (99,772 valid, 99.8%) | 5,924 |
| **Los Angeles** | 100,000 | 28 | 2020-01-01 to 2020-12-31 | 126 | Yes (99,549 valid, 99.5%) | Yes (99,549 valid, 99.5%) | 546,515 |
| **San Francisco** | 100,000 | 36 | 2018-01-01 to 2026-10-22 | 48 | Yes (41,705 valid, 41.7%) | Yes (41,705 valid, 41.7%) | 1,135,079 |
| **UK** | 174,216 | 12 | 2024-01 to 2024-06 | 14 | Yes (174,216 valid, 100.0%) | Yes (174,216 valid, 100.0%) | 48,312 |

---

## 3. Deep-Dive Regional Inspection

### 3.1 Chicago Crime Dataset
* **Official Source:** City of Chicago Open Data Portal (`https://data.cityofchicago.org/`, Socrata API endpoint `ijzp-q8t2`).
* **Raw File:** `data/raw/chicago/chicago_crimes_raw.json`
* **Volume:** 100,000 records | 31 features | 0 duplicate record IDs.
* **Date Range:** 2026-05-18 to 2026-10-15 (High-density temporal slice).
* **Target Category Feature:** `primary_type` (31 distinct categories).
  * **Top Categories:** `THEFT` (21,264; 21.3%), `BATTERY` (18,707; 18.7%), `CRIMINAL DAMAGE` (11,244; 11.2%), `ASSAULT` (9,258; 9.3%), `MOTOR VEHICLE THEFT` (7,836; 7.8%), `OTHER OFFENSE` (5,702; 5.7%), `DECEPTIVE PRACTICE` (5,334; 5.3%), `ROBBERY` (4,498; 4.5%), `WEAPONS VIOLATION` (4,233; 4.2%), `BURGLARY` (3,411; 3.4%).
  * **Minority/Long-Tail Classes:** `NON-CRIMINAL` (3), `HUMAN TRAFFICKING` (8), `PUBLIC INDECENCY` (10), `ARSON` (203). High class imbalance (ratio > 7000:1).
* **Geospatial Properties:**
  * Coordinates present in `latitude` and `longitude` fields (99,772 valid non-zero points, 99.8% spatial completeness).
  * Projected coordinates present: `x_coordinate`, `y_coordinate`.
  * Spatial Boundary: Lat [41.6446, 42.0225], Lon [-87.9274, -87.5245].
  * Administrative regions: `beat`, `district`, `ward`, `community_area`, Socrata computed regions.
* **Temporal Properties:** `date` field contains full ISO timestamp (`YYYY-MM-DDTHH:MM:SS.000`), allowing extraction of Hour of Day, Day of Week, Month, and Weekend indicators.
* **Other Contextual Features:** `arrest` (boolean), `domestic` (boolean), `location_description` (e.g., Street, Residence, Sidewalk), `fbi_code`, `iucr`.

---

### 3.2 Los Angeles Crime Dataset
* **Official Source:** Los Angeles Open Data Portal (`https://data.lacity.gov/`, Socrata API endpoint `2nrs-mtv8`).
* **Raw File:** `data/raw/los_angeles/los_angeles_crimes_raw.json`
* **Volume:** 100,000 records | 28 features | 0 duplicate record IDs.
* **Date Range:** 2020-01-01 to 2020-12-31.
* **Target Category Feature:** `crm_cd_desc` (126 distinct categories) / `crm_cd` (numeric code).
  * **Top Categories:** `VEHICLE - STOLEN` (10,741; 10.7%), `BATTERY - SIMPLE ASSAULT` (7,640; 7.6%), `BURGLARY` (6,439; 6.4%), `BURGLARY FROM VEHICLE` (6,143; 6.1%), `VANDALISM - FELONY` (6,039; 6.0%), `THEFT OF IDENTITY` (5,836; 5.8%), `INTIMATE PARTNER - SIMPLE ASSAULT` (5,700; 5.7%), `ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT` (5,124; 5.1%), `ROBBERY` (3,391; 3.4%), `THEFT PLAIN - PETTY ($950 & UNDER)` (3,303; 3.3%).
  * **Minority/Long-Tail Classes:** 126 classes with severe long-tail distribution (e.g., specialized offenses occurring < 5 times). Grouping into top $K$ (e.g., top 15-20) or hierarchical clustering will be necessary in preprocessing.
* **Geospatial Properties:**
  * Coordinates present in `lat` and `lon` fields (99,549 valid coordinates, 99.5% spatial completeness; only 451 records default to 0.0).
  * Spatial Boundary: Lat [33.7065, 34.3293], Lon [-118.6676, -118.1560].
  * Administrative regions: `area` (Division ID 1-21), `area_name` (e.g., Central, Pacific, 77th Street), `rpt_dist_no` (Reporting District).
* **Temporal Properties:** `date_occ` (Date occurred) and `time_occ` (Military 24-hr integer time, e.g., `0330`, `2145`), enabling hourly and cyclical time transformations.
* **Other Contextual Features:** Unique victim demographic variables (`vict_age`, `vict_sex`, `vict_descent`), premises code (`premis_desc`), weapon information (`weapon_desc`), and case disposition status (`status_desc`).

---

### 3.3 San Francisco Crime Dataset
* **Official Source:** DataSF (`https://data.sfgov.org/`, Socrata API endpoint `wg3w-h783`).
* **Raw File:** `data/raw/san_francisco/san_francisco_crimes_raw.json`
* **Volume:** 100,000 records | 36 features | 1,177 duplicate row IDs (associated with multi-charge incidents or amendatory filings).
* **Date Range:** 2018-01-01 to 2026-10-22.
* **Target Category Feature:** `incident_category` (48 distinct categories).
  * **Top Categories:** `Larceny Theft` (39,097; 39.1%), `Recovered Vehicle` (13,799; 13.8%), `Lost Property` (7,983; 8.0%), `Malicious Mischief` (5,499; 5.5%), `Assault` (3,799; 3.8%), `Motor Vehicle Theft` (3,674; 3.7%), `Burglary` (3,607; 3.6%), `Fraud` (3,158; 3.2%), `Non-Criminal` (3,059; 3.1%), `Robbery` (2,168; 2.2%).
  * **Missing Target:** 363 records have null `incident_category`.
* **Geospatial Properties:**
  * Coordinates present in `latitude` and `longitude` fields. Valid non-null, non-zero coordinates: 41,705 (41.7%).
  * Reason for missing coordinates: Online citizen filings (`filed_online == True`), non-location property recoveries, and privacy suppression for sensitive offenses.
  * Spatial Boundary (Valid subset): Lat [37.7079, 37.8301], Lon [-122.5119, -122.3663].
  * Administrative regions: `police_district` (e.g., Central, Mission, Tenderloin), `analysis_neighborhood`, `supervisor_district`, `intersection`.
* **Temporal Properties:** `incident_datetime`, `incident_date`, `incident_time`, `incident_day_of_week`, `incident_year`.
* **Other Contextual Features:** `report_type_description` (Initial vs Coplogic online), `filed_online` (boolean), `resolution` (Open/Active, Cite/Arrested), `incident_subcategory`.

---

### 3.4 UK Crime Dataset
* **Official Source:** UK Police Data Portal (`https://data.police.uk/`, API endpoint `crimes-street/all-crime`).
* **Raw File:** `data/raw/uk/uk_crimes_raw.json`
* **Volume:** 174,216 records | 12 features | 0 duplicate record IDs.
* **Date Range:** 2024-01 to 2024-06 (6-month period across 14 major police jurisdictions).
* **Target Category Feature:** `category` (14 standardized Home Office categories).
  * **Categories & Distribution:** `violent-crime` (40,588; 23.3%), `anti-social-behaviour` (24,156; 13.9%), `theft-from-the-person` (20,029; 11.5%), `other-theft` (19,867; 11.4%), `shoplifting` (16,824; 9.7%), `public-order` (15,417; 8.8%), `vehicle-crime` (12,398; 7.1%), `drugs` (6,552; 3.8%), `robbery` (5,917; 3.4%), `burglary` (5,595; 3.2%), `criminal-damage-arson` (4,411; 2.5%), `other-crime` (1,299; 0.7%), `possession-of-weapons` (668; 0.4%), `bicycle-theft` (495; 0.3%).
* **Geospatial Properties:**
  * Coordinates present in nested `location.latitude` and `location.longitude` (174,216 valid coordinate pairs, 100.0% spatial completeness).
  * Spatial Boundary: Lat [50.8954, 54.9921], Lon [-3.2019, -0.0323].
  * Geographic Reference: Anonymized snap-to-street centroids (`street.id`, `street.name`, e.g., "On or near Nightclub", "On or near Shopping Area").
* **Temporal Properties:** `month` (`YYYY-MM`). Note: UK Police API masks exact timestamps to Protect individual anonymity; temporal features are month-level and seasonal.
* **Other Contextual Features:** `location_type` (Force), `context`, `outcome_status.category` (e.g., Under investigation, Unable to prosecute suspect, Action to be taken by another organisation).

---

## 4. Key Cross-Dataset Differences

| Dimension | Chicago | Los Angeles | San Francisco | UK |
| :--- | :--- | :--- | :--- | :--- |
| **Data Provider** | Chicago Police Dept / Socrata | LAPD / Socrata | SFPD / DataSF | UK Home Office Police API |
| **Categorization Scheme** | IUCR / Illinois UCR (31 types) | LAPD Penal Code (126 types) | SFPD Incident Categories (48 types) | Home Office 14 Standard Types |
| **Class Granularity** | High-level statutory groupings | Highly granular offense codes | Mixed incident categories | Broad standardized offense types |
| **Temporal Granularity** | Minute-level (`YYYY-MM-DD HH:MM`) | Minute-level (`date_occ` + military `time_occ`) | Minute-level (`YYYY-MM-DD HH:MM`) | Monthly (`YYYY-MM`) |
| **Spatial Completeness** | 99.8% | 99.5% | 41.7% (Online reports omit coords) | 100.0% (Snapped to street points) |
| **Demographic Features** | None | Victim Age, Sex, Descent | None | None |
| **Investigative Outcome** | Arrest Made (True/False) | Status Code (Adult/Juv Arrest, Inv) | Resolution (Arrest, Open) | Formal Criminal Justice Outcome |

---

## 5. Machine Learning Feasibility & Task Recommendations

### Task 1: Crime Category Classification
* **Chicago:** Highly suitable. Excellent feature set (`location_description`, `domestic`, `beat`, `district`, `hour`, `day_of_week`, `month`, `latitude`, `longitude`). Recommend top 15 categories representing 95%+ of volume.
* **Los Angeles:** Highly suitable. Rich tabular predictors (`premis_desc`, `weapon_desc`, `vict_age`, `vict_sex`, `vict_descent`, `area_name`, `hour`, `day_of_week`, `lat`, `lon`). Recommend grouping 126 classes to top 20 or parent penal codes.
* **San Francisco:** Highly suitable on records with valid categories. Features: `police_district`, `analysis_neighborhood`, `hour`, `day_of_week`, `month`, `filed_online`, `lat`, `lon`.
* **UK:** Suitable with balanced 14 classes. Features: `location_type`, `street_name` / POI anchor description, `month`, `latitude`, `longitude`.

### Task 2: Spatial Hotspot Classification
* **Chicago:** **Fully Feasible**. 99,772 coordinates form dense spatial clusters across Chicago's 77 community areas and 22 police districts. Suitable for spatial density grid discretization (e.g., $k$-means spatial bins, H3 hex bins, or high/medium/low crime density zones).
* **Los Angeles:** **Fully Feasible**. 99,549 coordinates cover 21 LAPD divisions with high density.
* **San Francisco:** **Feasible with Filtering**. Restricting to records with valid geographic coordinates (41,705 records) yields dense spatial coverage over the SF peninsula.
* **UK:** **Fully Feasible**. 174,216 valid coordinates across urban police clusters (London, Birmingham, Leeds, Liverpool, Bristol, Newcastle, Sheffield, Nottingham, Leicester, Southampton, Cardiff).

---

## 6. Verification and Compliance Checklist

- [x] Four independent datasets acquired programmatically into dedicated raw directories.
- [x] Raw files preserved unedited without alteration.
- [x] Full structural inspection executed (columns, types, missing values, duplicates, date ranges, bounding boxes).
- [x] Latitude and longitude availability verified for each dataset.
- [x] Class distributions and imbalance documented.
- [x] Dataset summary table saved to `results/reports/dataset_summary.csv`.
- [x] Inspection report saved to `results/reports/dataset_inspection.md`.
- [x] **No ML models trained** (Stage strictly bounded to Part 1).
