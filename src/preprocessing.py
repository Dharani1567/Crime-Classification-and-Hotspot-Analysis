"""
src/preprocessing.py
Independent Preprocessing Pipelines for the Four Crime Datasets:
1. Chicago Crime Dataset
2. Los Angeles Crime Dataset
3. San Francisco Crime Dataset
4. UK Crime Dataset

Adheres strictly to project constraints:
- Four independent pipelines (no merging)
- Removes exact duplicates
- Handles missing values systematically per field type
- Standardizes column names
- Converts and validates datetime fields
- Validates geographic coordinates within regional bounding boxes
- Identifies rare crime categories and computes class imbalance
- Saves cleaned datasets to data/processed/<region>/
"""

import os
import json
import numpy as np
import pandas as pd

# Regional Bounding Boxes (for coordinate validity checking)
BBOXES = {
    'chicago': {'lat_min': 41.60, 'lat_max': 42.10, 'lon_min': -88.00, 'lon_max': -87.45},
    'los_angeles': {'lat_min': 33.65, 'lat_max': 34.40, 'lon_min': -118.75, 'lon_max': -118.10},
    'san_francisco': {'lat_min': 37.65, 'lat_max': 37.85, 'lon_min': -122.55, 'lon_max': -122.30},
    'uk': {'lat_min': 49.80, 'lat_max': 58.70, 'lon_min': -8.00, 'lon_max': 2.00}
}

def clean_chicago():
    print("\n" + "="*50)
    print("PREPROCESSING PIPELINE: CHICAGO")
    print("="*50)
    input_path = 'data/raw/chicago/chicago_crimes_raw.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    initial_rows = len(df)
    print(f"Loaded raw records: {initial_rows:,}")

    # 1. Standardize column names (lowercase with underscores)
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Drop duplicates
    if 'id' in df.columns:
        dups = df.duplicated(subset=['id']).sum()
        df = df.drop_duplicates(subset=['id'])
    else:
        dups = df.duplicated().sum()
        df = df.drop_duplicates()
    print(f"Removed duplicate records: {dups:,}")

    # 3. Datetime conversion
    df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_of_week_num'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)

    # 4. Geographic coordinates validation
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    bb = BBOXES['chicago']
    valid_geo = (
        (df['latitude'] >= bb['lat_min']) & (df['latitude'] <= bb['lat_max']) &
        (df['longitude'] >= bb['lon_min']) & (df['longitude'] <= bb['lon_max'])
    )
    df['has_valid_coordinates'] = valid_geo.astype(int)

    # 5. Missing values imputation
    df['location_description'] = df['location_description'].fillna('OTHER/UNKNOWN').astype(str).str.strip().str.upper()
    df['district'] = df['district'].fillna('UNKNOWN').astype(str)
    df['ward'] = df['ward'].fillna('-1').astype(str)
    df['community_area'] = df['community_area'].fillna('-1').astype(str)
    df['beat'] = df['beat'].fillna('UNKNOWN').astype(str)
    df['arrest'] = df['arrest'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)
    df['domestic'] = df['domestic'].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)

    # 6. Crime Category Target Standardization
    df = df.dropna(subset=['primary_type'])
    df['crime_category'] = df['primary_type'].str.strip().str.upper()

    # Identify rare categories (threshold: < 50 occurrences in 100k)
    cat_counts = df['crime_category'].value_counts()
    rare_cats = cat_counts[cat_counts < 50].index.tolist()
    print(f"Total crime categories: {len(cat_counts)}. Rare categories (<50 occurrences): {len(rare_cats)}")
    df['is_rare_category'] = df['crime_category'].isin(rare_cats).astype(int)

    # Select sanitized columns for processed output
    keep_cols = [
        'id', 'datetime', 'hour', 'day_of_week', 'day_of_week_num', 'month', 'is_weekend',
        'latitude', 'longitude', 'has_valid_coordinates',
        'district', 'ward', 'community_area', 'beat', 'location_description',
        'arrest', 'domestic', 'crime_category', 'is_rare_category'
    ]
    processed_df = df[keep_cols].copy()
    processed_df = processed_df.sort_values('datetime').reset_index(drop=True)

    out_file = 'data/processed/chicago/chicago_processed.csv'
    processed_df.to_csv(out_file, index=False)
    print(f"Saved processed Chicago dataset ({len(processed_df):,} rows, {len(processed_df.columns)} cols) to {out_file}")
    
    return {
        'initial_rows': initial_rows,
        'final_rows': len(processed_df),
        'duplicates_removed': int(dups),
        'valid_coords_pct': float(processed_df['has_valid_coordinates'].mean() * 100),
        'total_categories': len(cat_counts),
        'rare_categories_count': len(rare_cats)
    }

def clean_los_angeles():
    print("\n" + "="*50)
    print("PREPROCESSING PIPELINE: LOS ANGELES")
    print("="*50)
    input_path = 'data/raw/los_angeles/los_angeles_crimes_raw.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    initial_rows = len(df)
    print(f"Loaded raw records: {initial_rows:,}")

    # 1. Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Drop duplicates
    if 'dr_no' in df.columns:
        dups = df.duplicated(subset=['dr_no']).sum()
        df = df.drop_duplicates(subset=['dr_no'])
    else:
        dups = df.duplicated().sum()
        df = df.drop_duplicates()
    print(f"Removed duplicate records: {dups:,}")

    # 3. Datetime conversion
    # date_occ is YYYY-MM-DD, time_occ is 24-hr military string e.g. '0845', '30'
    df['time_str'] = df['time_occ'].astype(str).str.zfill(4)
    df['hour'] = pd.to_numeric(df['time_str'].str[:2], errors='coerce').fillna(0).astype(int)
    df['minute'] = pd.to_numeric(df['time_str'].str[2:], errors='coerce').fillna(0).astype(int)
    
    df['date_only'] = pd.to_datetime(df['date_occ'], errors='coerce')
    df = df.dropna(subset=['date_only'])
    df['datetime'] = df['date_only'] + pd.to_timedelta(df['hour'], unit='h') + pd.to_timedelta(df['minute'], unit='m')
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_of_week_num'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)

    # 4. Geographic coordinates validation
    df['latitude'] = pd.to_numeric(df['lat'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['lon'], errors='coerce')
    bb = BBOXES['los_angeles']
    valid_geo = (
        (df['latitude'] >= bb['lat_min']) & (df['latitude'] <= bb['lat_max']) &
        (df['longitude'] >= bb['lon_min']) & (df['longitude'] <= bb['lon_max'])
    )
    df['has_valid_coordinates'] = valid_geo.astype(int)

    # 5. Missing values and domain features
    df['area_name'] = df['area_name'].fillna('UNKNOWN').astype(str).str.strip()
    df['area'] = df['area'].fillna('-1').astype(str)
    df['rpt_dist_no'] = df['rpt_dist_no'].fillna('-1').astype(str)
    df['premis_desc'] = df['premis_desc'].fillna('UNKNOWN/OTHER').astype(str).str.strip()
    df['weapon_desc'] = df['weapon_desc'].fillna('NO WEAPON REPORTED').astype(str).str.strip()
    
    # Victim demographics
    df['vict_age'] = pd.to_numeric(df['vict_age'], errors='coerce').fillna(-1).astype(int)
    df['vict_sex'] = df['vict_sex'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
    df['vict_sex'] = df['vict_sex'].replace({'': 'UNKNOWN', 'H': 'UNKNOWN', 'X': 'OTHER'})
    df['vict_descent'] = df['vict_descent'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
    df['status_desc'] = df['status_desc'].fillna('UNKNOWN').astype(str).str.strip()

    # 6. Crime Category Target Standardization
    df = df.dropna(subset=['crm_cd_desc'])
    df['crime_category'] = df['crm_cd_desc'].str.strip().str.upper()

    cat_counts = df['crime_category'].value_counts()
    rare_cats = cat_counts[cat_counts < 50].index.tolist()
    print(f"Total crime categories: {len(cat_counts)}. Rare categories (<50 occurrences): {len(rare_cats)}")
    df['is_rare_category'] = df['crime_category'].isin(rare_cats).astype(int)

    keep_cols = [
        'dr_no', 'datetime', 'hour', 'day_of_week', 'day_of_week_num', 'month', 'is_weekend',
        'latitude', 'longitude', 'has_valid_coordinates',
        'area', 'area_name', 'rpt_dist_no', 'premis_desc', 'weapon_desc',
        'vict_age', 'vict_sex', 'vict_descent', 'status_desc',
        'crime_category', 'is_rare_category'
    ]
    processed_df = df[keep_cols].copy()
    processed_df = processed_df.sort_values('datetime').reset_index(drop=True)

    out_file = 'data/processed/los_angeles/los_angeles_processed.csv'
    processed_df.to_csv(out_file, index=False)
    print(f"Saved processed LA dataset ({len(processed_df):,} rows, {len(processed_df.columns)} cols) to {out_file}")

    return {
        'initial_rows': initial_rows,
        'final_rows': len(processed_df),
        'duplicates_removed': int(dups),
        'valid_coords_pct': float(processed_df['has_valid_coordinates'].mean() * 100),
        'total_categories': len(cat_counts),
        'rare_categories_count': len(rare_cats)
    }

def clean_san_francisco():
    print("\n" + "="*50)
    print("PREPROCESSING PIPELINE: SAN FRANCISCO")
    print("="*50)
    input_path = 'data/raw/san_francisco/san_francisco_crimes_raw.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    initial_rows = len(df)
    print(f"Loaded raw records: {initial_rows:,}")

    # 1. Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Drop duplicates
    if 'row_id' in df.columns:
        dups = df.duplicated(subset=['row_id']).sum()
        df = df.drop_duplicates(subset=['row_id'])
    else:
        dups = df.duplicated().sum()
        df = df.drop_duplicates()
    print(f"Removed duplicate records: {dups:,}")

    # 3. Datetime conversion
    df['datetime'] = pd.to_datetime(df['incident_datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_of_week_num'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)

    # 4. Geographic coordinates validation
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    bb = BBOXES['san_francisco']
    valid_geo = (
        (df['latitude'] >= bb['lat_min']) & (df['latitude'] <= bb['lat_max']) &
        (df['longitude'] >= bb['lon_min']) & (df['longitude'] <= bb['lon_max'])
    )
    df['has_valid_coordinates'] = valid_geo.astype(int)

    # 5. Missing values and domain features
    df['police_district'] = df['police_district'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
    df['analysis_neighborhood'] = df['analysis_neighborhood'].fillna('UNKNOWN').astype(str).str.strip()
    df['supervisor_district'] = df['supervisor_district'].fillna('-1').astype(str)
    df['filed_online'] = df['filed_online'].fillna(False).astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)
    df['report_type_description'] = df['report_type_description'].fillna('UNKNOWN').astype(str).str.strip()
    df['resolution'] = df['resolution'].fillna('UNKNOWN').astype(str).str.strip()

    # 6. Crime Category Target Standardization
    df = df.dropna(subset=['incident_category'])
    df['crime_category'] = df['incident_category'].str.strip().str.upper()

    cat_counts = df['crime_category'].value_counts()
    rare_cats = cat_counts[cat_counts < 50].index.tolist()
    print(f"Total crime categories: {len(cat_counts)}. Rare categories (<50 occurrences): {len(rare_cats)}")
    df['is_rare_category'] = df['crime_category'].isin(rare_cats).astype(int)

    keep_cols = [
        'row_id', 'datetime', 'hour', 'day_of_week', 'day_of_week_num', 'month', 'is_weekend',
        'latitude', 'longitude', 'has_valid_coordinates',
        'police_district', 'analysis_neighborhood', 'supervisor_district',
        'filed_online', 'report_type_description', 'resolution',
        'crime_category', 'is_rare_category'
    ]
    processed_df = df[keep_cols].copy()
    processed_df = processed_df.sort_values('datetime').reset_index(drop=True)

    out_file = 'data/processed/san_francisco/san_francisco_processed.csv'
    processed_df.to_csv(out_file, index=False)
    print(f"Saved processed SF dataset ({len(processed_df):,} rows, {len(processed_df.columns)} cols) to {out_file}")

    return {
        'initial_rows': initial_rows,
        'final_rows': len(processed_df),
        'duplicates_removed': int(dups),
        'valid_coords_pct': float(processed_df['has_valid_coordinates'].mean() * 100),
        'total_categories': len(cat_counts),
        'rare_categories_count': len(rare_cats)
    }

def clean_uk():
    print("\n" + "="*50)
    print("PREPROCESSING PIPELINE: UK")
    print("="*50)
    input_path = 'data/raw/uk/uk_crimes_raw.json'
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    initial_rows = len(data)
    print(f"Loaded raw records: {initial_rows:,}")

    # Flatten nested UK structure
    records = []
    for r in data:
        loc = r.get('location') or {}
        st = loc.get('street') or {}
        outcome = r.get('outcome_status') or {}
        rec = {
            'id': str(r.get('id') or ''),
            'month': str(r.get('month') or ''),
            'category': str(r.get('category') or ''),
            'location_type': str(r.get('location_type') or 'Force'),
            'latitude': loc.get('latitude'),
            'longitude': loc.get('longitude'),
            'street_id': str(st.get('id') or ''),
            'street_name': str(st.get('name') or 'UNKNOWN'),
            'outcome_category': str(outcome.get('category') or 'NO_OUTCOME_RECORDED')
        }
        records.append(rec)
    df = pd.DataFrame(records)

    # 1. Standardize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Drop duplicates
    # For records with non-empty id, check duplication
    has_id_mask = df['id'] != ''
    dups_id = df[has_id_mask].duplicated(subset=['id']).sum()
    df = pd.concat([
        df[has_id_mask].drop_duplicates(subset=['id']),
        df[~has_id_mask]
    ]).reset_index(drop=True)
    # Check overall exact duplicates
    exact_dups = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    total_dups = dups_id + exact_dups
    print(f"Removed duplicate records: {total_dups:,}")

    # 3. Datetime conversion
    df['datetime'] = pd.to_datetime(df['month'], format='%Y-%m', errors='coerce')
    df = df.dropna(subset=['datetime'])
    df['month_num'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year

    # 4. Geographic coordinates validation
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    bb = BBOXES['uk']
    valid_geo = (
        (df['latitude'] >= bb['lat_min']) & (df['latitude'] <= bb['lat_max']) &
        (df['longitude'] >= bb['lon_min']) & (df['longitude'] <= bb['lon_max'])
    )
    df['has_valid_coordinates'] = valid_geo.astype(int)

    # 5. Missing values and domain features
    df['location_type'] = df['location_type'].fillna('Force').astype(str).str.strip()
    df['street_name'] = df['street_name'].fillna('UNKNOWN').astype(str).str.strip().str.upper()
    df['outcome_category'] = df['outcome_category'].fillna('NO_OUTCOME_RECORDED').astype(str).str.strip()

    # 6. Crime Category Target Standardization
    df = df.dropna(subset=['category'])
    df['crime_category'] = df['category'].str.strip().str.lower()

    cat_counts = df['crime_category'].value_counts()
    rare_cats = cat_counts[cat_counts < 50].index.tolist()
    print(f"Total crime categories: {len(cat_counts)}. Rare categories (<50 occurrences): {len(rare_cats)}")
    df['is_rare_category'] = df['crime_category'].isin(rare_cats).astype(int)

    keep_cols = [
        'id', 'datetime', 'year', 'month_num',
        'latitude', 'longitude', 'has_valid_coordinates',
        'location_type', 'street_name', 'outcome_category',
        'crime_category', 'is_rare_category'
    ]
    processed_df = df[keep_cols].copy()
    processed_df = processed_df.sort_values('datetime').reset_index(drop=True)

    out_file = 'data/processed/uk/uk_processed.csv'
    processed_df.to_csv(out_file, index=False)
    print(f"Saved processed UK dataset ({len(processed_df):,} rows, {len(processed_df.columns)} cols) to {out_file}")

    return {
        'initial_rows': initial_rows,
        'final_rows': len(processed_df),
        'duplicates_removed': int(total_dups),
        'valid_coords_pct': float(processed_df['has_valid_coordinates'].mean() * 100),
        'total_categories': len(cat_counts),
        'rare_categories_count': len(rare_cats)
    }

def run_all_preprocessing():
    os.makedirs('data/processed/chicago', exist_ok=True)
    os.makedirs('data/processed/los_angeles', exist_ok=True)
    os.makedirs('data/processed/san_francisco', exist_ok=True)
    os.makedirs('data/processed/uk', exist_ok=True)

    metrics = {}
    metrics['Chicago'] = clean_chicago()
    metrics['Los Angeles'] = clean_los_angeles()
    metrics['San Francisco'] = clean_san_francisco()
    metrics['UK'] = clean_uk()

    with open('results/reports/preprocessing_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print("\nAll preprocessing completed. Metrics saved to results/reports/preprocessing_metrics.json")

if __name__ == '__main__':
    run_all_preprocessing()
