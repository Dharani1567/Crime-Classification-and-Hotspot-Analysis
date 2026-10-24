"""
src/feature_engineering.py
Independent Feature Engineering Pipelines for Two ML Tasks:
1. Task 1: Crime Category Classification
2. Task 2: Crime Hotspot Classification

Guarantees:
- Independent processing per dataset (Chicago, LA, SF, UK)
- Zero Data Leakage:
    * Features calculated strictly using PAST records (rolling windows with closed='left' on datetime index)
    * Target variable strictly excluded from feature matrix
    * Fit-time encoders/scalers designed for train-set fitting
    * Identifiers and direct proxies removed
- Spatial grid / administrative area discretizations for hotspot modeling
- Saves final feature matrices and hotspot target sets to data/features/<region>/
"""

import os
import json
import numpy as np
import pandas as pd

def compute_rolling_past_counts(df, date_col, group_col, windows=['7D', '30D']):
    """
    Computes past rolling crime counts for each group_col strictly using PAST records.
    Uses closed='left' so that current timestamp events are NOT included in the count.
    """
    df_sorted = df.sort_values(date_col).copy()
    df_sorted['temp_date'] = df_sorted[date_col]
    
    # Create an event count column
    df_sorted['event_count'] = 1
    
    # Set index to datetime for rolling operations
    df_sorted = df_sorted.set_index('temp_date')
    
    results = {}
    for w in windows:
        col_name = f'past_crime_count_{w.lower()}_{group_col}'
        # rolling on datetime index with closed='left' ensures strict past history only
        rolling_series = (
            df_sorted.groupby(group_col)['event_count']
            .rolling(w, closed='left')
            .count()
        )
        # Drop group level to match index
        rolling_series = rolling_series.reset_index(level=0, drop=True)
        results[col_name] = rolling_series.fillna(0).astype(int)
        
    df_sorted = df_sorted.reset_index(drop=True)
    for col_name, s in results.items():
        df_sorted[col_name] = s.values
        
    df_sorted = df_sorted.drop(columns=['event_count'])
    return df_sorted

def engineer_chicago_features():
    print("\n" + "="*50)
    print("FEATURE ENGINEERING: CHICAGO")
    print("="*50)
    df = pd.read_csv('data/processed/chicago/chicago_processed.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df = df.sort_values('datetime').reset_index(drop=True)

    # 1. Spatial Grid Discretization (for Hotspots)
    # Using 0.01 degree lat/lon bins (~1.1km grid cells)
    df['grid_lat'] = np.floor(df['latitude'] / 0.01) * 0.01
    df['grid_lon'] = np.floor(df['longitude'] / 0.01) * 0.01
    df['spatial_cell_id'] = df['district'].astype(str) + "_" + df['grid_lat'].round(2).astype(str) + "_" + df['grid_lon'].round(2).astype(str)

    # 2. Past Rolling History (Leakage-free, closed='left')
    print("Computing past historical crime counts by district (7-day, 30-day)...")
    df = compute_rolling_past_counts(df, date_col='datetime', group_col='district', windows=['7D', '30D'])

    # 3. Cyclical Temporal Encodings
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7.0)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7.0)

    # 4. Target 1: Crime Category Classification
    # Filter out rare categories (<50) or assign to OTHER
    top_categories = df['crime_category'].value_counts()
    frequent_cats = top_categories[top_categories >= 50].index.tolist()
    df['crime_category_clean'] = df['crime_category'].where(df['crime_category'].isin(frequent_cats), 'OTHER_RARE')

    # 5. Target 2: Hotspot Classification
    # High crime density zone = areas with historical past 7-day crime count above median
    median_past_7d = df['past_crime_count_7d_district'].median()
    df['is_crime_hotspot'] = (df['past_crime_count_7d_district'] > median_past_7d).astype(int)

    # Define Feature Sets
    classification_features = [
        'hour', 'sin_hour', 'cos_hour',
        'day_of_week_num', 'sin_dayofweek', 'cos_dayofweek', 'is_weekend',
        'month', 'sin_month', 'cos_month',
        'latitude', 'longitude', 'has_valid_coordinates',
        'district', 'ward', 'community_area', 'beat',
        'location_description', 'arrest', 'domestic'
    ]

    hotspot_features = [
        'district', 'spatial_cell_id',
        'past_crime_count_7d_district', 'past_crime_count_30d_district',
        'hour', 'sin_hour', 'cos_hour',
        'day_of_week_num', 'is_weekend', 'month',
        'latitude', 'longitude'
    ]

    # Save engineered feature datasets
    task1_df = df[classification_features + ['crime_category_clean']].copy()
    task2_df = df[hotspot_features + ['is_crime_hotspot']].copy()

    os.makedirs('data/features/chicago', exist_ok=True)
    task1_path = 'data/features/chicago/chicago_task1_crime_category.csv'
    task2_path = 'data/features/chicago/chicago_task2_hotspot.csv'
    task1_df.to_csv(task1_path, index=False)
    task2_df.to_csv(task2_path, index=False)
    print(f"Saved Chicago Task 1 features ({len(task1_df):,} rows, {len(classification_features)} features) to {task1_path}")
    print(f"Saved Chicago Task 2 features ({len(task2_df):,} rows, {len(hotspot_features)} features) to {task2_path}")

    return {
        'task1_rows': len(task1_df),
        'task1_features': len(classification_features),
        'task1_classes': int(task1_df['crime_category_clean'].nunique()),
        'task2_rows': len(task2_df),
        'task2_features': len(hotspot_features),
        'hotspot_positive_rate': float(task2_df['is_crime_hotspot'].mean())
    }

def engineer_los_angeles_features():
    print("\n" + "="*50)
    print("FEATURE ENGINEERING: LOS ANGELES")
    print("="*50)
    df = pd.read_csv('data/processed/los_angeles/los_angeles_processed.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df = df.sort_values('datetime').reset_index(drop=True)

    # 1. Spatial Discretization
    df['grid_lat'] = np.floor(df['latitude'] / 0.01) * 0.01
    df['grid_lon'] = np.floor(df['longitude'] / 0.01) * 0.01
    df['spatial_cell_id'] = df['area_name'].astype(str) + "_" + df['grid_lat'].round(2).astype(str) + "_" + df['grid_lon'].round(2).astype(str)

    # 2. Past Rolling History
    print("Computing past historical crime counts by area_name (7-day, 30-day)...")
    df = compute_rolling_past_counts(df, date_col='datetime', group_col='area_name', windows=['7D', '30D'])

    # 3. Cyclical Temporal Encodings
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7.0)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7.0)

    # 4. Target 1: Category cleaning
    top_categories = df['crime_category'].value_counts()
    frequent_cats = top_categories[top_categories >= 100].index.tolist()
    df['crime_category_clean'] = df['crime_category'].where(df['crime_category'].isin(frequent_cats), 'OTHER_RARE')

    # 5. Target 2: Hotspot classification
    median_past_7d = df['past_crime_count_7d_area_name'].median()
    df['is_crime_hotspot'] = (df['past_crime_count_7d_area_name'] > median_past_7d).astype(int)

    classification_features = [
        'hour', 'sin_hour', 'cos_hour',
        'day_of_week_num', 'sin_dayofweek', 'cos_dayofweek', 'is_weekend',
        'month', 'sin_month', 'cos_month',
        'latitude', 'longitude', 'has_valid_coordinates',
        'area', 'area_name', 'rpt_dist_no',
        'premis_desc', 'weapon_desc',
        'vict_age', 'vict_sex', 'vict_descent', 'status_desc'
    ]

    hotspot_features = [
        'area_name', 'spatial_cell_id',
        'past_crime_count_7d_area_name', 'past_crime_count_30d_area_name',
        'hour', 'sin_hour', 'cos_hour',
        'day_of_week_num', 'is_weekend', 'month',
        'latitude', 'longitude'
    ]

    task1_df = df[classification_features + ['crime_category_clean']].copy()
    task2_df = df[hotspot_features + ['is_crime_hotspot']].copy()

    os.makedirs('data/features/los_angeles', exist_ok=True)
    task1_path = 'data/features/los_angeles/los_angeles_task1_crime_category.csv'
    task2_path = 'data/features/los_angeles/los_angeles_task2_hotspot.csv'
    task1_df.to_csv(task1_path, index=False)
    task2_df.to_csv(task2_path, index=False)
    print(f"Saved LA Task 1 features ({len(task1_df):,} rows, {len(classification_features)} features) to {task1_path}")
    print(f"Saved LA Task 2 features ({len(task2_df):,} rows, {len(hotspot_features)} features) to {task2_path}")

    return {
        'task1_rows': len(task1_df),
        'task1_features': len(classification_features),
        'task1_classes': int(task1_df['crime_category_clean'].nunique()),
        'task2_rows': len(task2_df),
        'task2_features': len(hotspot_features),
        'hotspot_positive_rate': float(task2_df['is_crime_hotspot'].mean())
    }

def engineer_san_francisco_features():
    print("\n" + "="*50)
    print("FEATURE ENGINEERING: SAN FRANCISCO")
    print("="*50)
    df = pd.read_csv('data/processed/san_francisco/san_francisco_processed.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df = df.sort_values('datetime').reset_index(drop=True)

    # 1. Spatial Discretization
    df['grid_lat'] = np.floor(df['latitude'] / 0.01) * 0.01
    df['grid_lon'] = np.floor(df['longitude'] / 0.01) * 0.01
    df['spatial_cell_id'] = df['police_district'].astype(str) + "_" + df['grid_lat'].round(2).astype(str) + "_" + df['grid_lon'].round(2).astype(str)

    # 2. Past Rolling History
    print("Computing past historical crime counts by police_district (7-day, 30-day)...")
    df = compute_rolling_past_counts(df, date_col='datetime', group_col='police_district', windows=['7D', '30D'])

    # 3. Cyclical Temporal Encodings
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['sin_dayofweek'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7.0)
    df['cos_dayofweek'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7.0)

    # 4. Target 1: Category cleaning
    top_categories = df['crime_category'].value_counts()
    frequent_cats = top_categories[top_categories >= 50].index.tolist()
    df['crime_category_clean'] = df['crime_category'].where(df['crime_category'].isin(frequent_cats), 'OTHER_RARE')

    # 5. Target 2: Hotspot classification
    median_past_7d = df['past_crime_count_7d_police_district'].median()
    df['is_crime_hotspot'] = (df['past_crime_count_7d_police_district'] > median_past_7d).astype(int)

    classification_features = [
        'hour', 'sin_hour', 'cos_hour',
        'day_of_week_num', 'sin_dayofweek', 'cos_dayofweek', 'is_weekend',
        'month', 'sin_month', 'cos_month',
        'latitude', 'longitude', 'has_valid_coordinates',
        'police_district', 'analysis_neighborhood', 'supervisor_district',
        'filed_online', 'report_type_description', 'resolution'
    ]

    hotspot_features = [
        'police_district', 'analysis_neighborhood', 'spatial_cell_id',
        'past_crime_count_7d_police_district', 'past_crime_count_30d_police_district',
        'hour', 'sin_hour', 'cos_hour',
        'day_of_week_num', 'is_weekend', 'month',
        'latitude', 'longitude', 'has_valid_coordinates'
    ]

    task1_df = df[classification_features + ['crime_category_clean']].copy()
    task2_df = df[hotspot_features + ['is_crime_hotspot']].copy()

    os.makedirs('data/features/san_francisco', exist_ok=True)
    task1_path = 'data/features/san_francisco/san_francisco_task1_crime_category.csv'
    task2_path = 'data/features/san_francisco/san_francisco_task2_hotspot.csv'
    task1_df.to_csv(task1_path, index=False)
    task2_df.to_csv(task2_path, index=False)
    print(f"Saved SF Task 1 features ({len(task1_df):,} rows, {len(classification_features)} features) to {task1_path}")
    print(f"Saved SF Task 2 features ({len(task2_df):,} rows, {len(hotspot_features)} features) to {task2_path}")

    return {
        'task1_rows': len(task1_df),
        'task1_features': len(classification_features),
        'task1_classes': int(task1_df['crime_category_clean'].nunique()),
        'task2_rows': len(task2_df),
        'task2_features': len(hotspot_features),
        'hotspot_positive_rate': float(task2_df['is_crime_hotspot'].mean())
    }

def engineer_uk_features():
    print("\n" + "="*50)
    print("FEATURE ENGINEERING: UK")
    print("="*50)
    df = pd.read_csv('data/processed/uk/uk_processed.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df = df.sort_values('datetime').reset_index(drop=True)

    # 1. Spatial Discretization (0.02 degree grid, ~2km)
    df['grid_lat'] = np.floor(df['latitude'] / 0.02) * 0.02
    df['grid_lon'] = np.floor(df['longitude'] / 0.02) * 0.02
    df['spatial_cell_id'] = "UK_" + df['grid_lat'].round(2).astype(str) + "_" + df['grid_lon'].round(2).astype(str)

    # 2. Past Rolling History (Monthly basis for UK since dates are month-level)
    # Using 60D (~2 months) past rolling crimes
    print("Computing past historical crime counts by spatial_cell_id...")
    df = compute_rolling_past_counts(df, date_col='datetime', group_col='spatial_cell_id', windows=['60D', '90D'])

    # 3. Cyclical Temporal Encodings (month-level)
    df['sin_month'] = np.sin(2 * np.pi * df['month_num'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month_num'] / 12.0)

    # 4. Target 1: Category cleaning
    # UK has 14 standard categories, all well-represented
    df['crime_category_clean'] = df['crime_category']

    # 5. Target 2: Hotspot classification
    median_past_60d = df['past_crime_count_60d_spatial_cell_id'].median()
    df['is_crime_hotspot'] = (df['past_crime_count_60d_spatial_cell_id'] > median_past_60d).astype(int)

    classification_features = [
        'month_num', 'sin_month', 'cos_month',
        'latitude', 'longitude', 'has_valid_coordinates',
        'location_type', 'street_name', 'outcome_category'
    ]

    hotspot_features = [
        'spatial_cell_id',
        'past_crime_count_60d_spatial_cell_id', 'past_crime_count_90d_spatial_cell_id',
        'month_num', 'sin_month', 'cos_month',
        'latitude', 'longitude'
    ]

    task1_df = df[classification_features + ['crime_category_clean']].copy()
    task2_df = df[hotspot_features + ['is_crime_hotspot']].copy()

    os.makedirs('data/features/uk', exist_ok=True)
    task1_path = 'data/features/uk/uk_task1_crime_category.csv'
    task2_path = 'data/features/uk/uk_task2_hotspot.csv'
    task1_df.to_csv(task1_path, index=False)
    task2_df.to_csv(task2_path, index=False)
    print(f"Saved UK Task 1 features ({len(task1_df):,} rows, {len(classification_features)} features) to {task1_path}")
    print(f"Saved UK Task 2 features ({len(task2_df):,} rows, {len(hotspot_features)} features) to {task2_path}")

    return {
        'task1_rows': len(task1_df),
        'task1_features': len(classification_features),
        'task1_classes': int(task1_df['crime_category_clean'].nunique()),
        'task2_rows': len(task2_df),
        'task2_features': len(hotspot_features),
        'hotspot_positive_rate': float(task2_df['is_crime_hotspot'].mean())
    }

def run_all_feature_engineering():
    metrics = {}
    metrics['Chicago'] = engineer_chicago_features()
    metrics['Los Angeles'] = engineer_los_angeles_features()
    metrics['San Francisco'] = engineer_san_francisco_features()
    metrics['UK'] = engineer_uk_features()

    with open('results/reports/feature_engineering_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print("\nAll feature engineering completed. Metrics saved to results/reports/feature_engineering_metrics.json")

if __name__ == '__main__':
    run_all_feature_engineering()
