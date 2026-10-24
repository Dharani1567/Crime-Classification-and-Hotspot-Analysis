"""
inspect_datasets.py
Comprehensive inspection and analysis of the four raw crime datasets:
1. Chicago (data/raw/chicago/chicago_crimes_raw.json)
2. Los Angeles (data/raw/los_angeles/los_angeles_crimes_raw.json)
3. San Francisco (data/raw/san_francisco/san_francisco_crimes_raw.json)
4. UK (data/raw/uk/uk_crimes_raw.json)

Generates:
- results/reports/dataset_summary.csv
- Detailed inspection metrics for results/reports/dataset_inspection.md
"""

import json
import pandas as pd
import numpy as np
import os

DATASETS = {
    'Chicago': {
        'path': 'data/raw/chicago/chicago_crimes_raw.json',
        'category_col': 'primary_type',
        'date_col': 'date',
        'lat_col': 'latitude',
        'lon_col': 'longitude',
        'id_col': 'id'
    },
    'Los Angeles': {
        'path': 'data/raw/los_angeles/los_angeles_crimes_raw.json',
        'category_col': 'crm_cd_desc',
        'date_col': 'date_occ',
        'lat_col': 'lat',
        'lon_col': 'lon',
        'id_col': 'dr_no'
    },
    'San Francisco': {
        'path': 'data/raw/san_francisco/san_francisco_crimes_raw.json',
        'category_col': 'incident_category',
        'date_col': 'incident_date',
        'lat_col': 'latitude',
        'lon_col': 'longitude',
        'id_col': 'row_id'
    },
    'UK': {
        'path': 'data/raw/uk/uk_crimes_raw.json',
        'category_col': 'category',
        'date_col': 'month',
        'lat_col': 'latitude',
        'lon_col': 'longitude',
        'id_col': 'id'
    }
}

def inspect_dataset(name, config):
    print(f"\n==================================================")
    print(f"INSPECTING: {name}")
    print(f"==================================================")
    
    with open(config['path'], 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Normalize nested structures if necessary (e.g. UK has nested location)
    if name == 'UK':
        flat_records = []
        for r in raw_data:
            rec = {
                'id': r.get('id'),
                'category': r.get('category'),
                'location_type': r.get('location_type'),
                'context': r.get('context'),
                'persistent_id': r.get('persistent_id'),
                'month': r.get('month'),
                'outcome_status_category': r.get('outcome_status', {}).get('category') if r.get('outcome_status') else None,
                'outcome_status_date': r.get('outcome_status', {}).get('date') if r.get('outcome_status') else None,
                'latitude': r.get('location', {}).get('latitude') if r.get('location') else None,
                'longitude': r.get('location', {}).get('longitude') if r.get('location') else None,
                'street_id': r.get('location', {}).get('street', {}).get('id') if r.get('location') else None,
                'street_name': r.get('location', {}).get('street', {}).get('name') if r.get('location') else None
            }
            flat_records.append(rec)
        df = pd.DataFrame(flat_records)
    else:
        df = pd.DataFrame(raw_data)
        
    num_records = len(df)
    num_cols = len(df.columns)
    
    # Check ID duplicates
    id_col = config.get('id_col')
    if id_col in df.columns:
        duplicate_ids = df[id_col].duplicated().sum()
    else:
        duplicate_ids = df.duplicated().sum()
        
    # Check nulls
    null_counts = df.isnull().sum()
    total_missing = null_counts.sum()
    
    # Date Range
    date_col = config['date_col']
    if date_col in df.columns:
        date_min = str(df[date_col].dropna().min())[:10]
        date_max = str(df[date_col].dropna().max())[:10]
        date_range_str = f"{date_min} to {date_max}"
    else:
        date_range_str = "N/A"
        
    # Crime Categories
    cat_col = config['category_col']
    if cat_col in df.columns:
        categories = df[cat_col].dropna().value_counts()
        num_categories = len(categories)
        top_categories = categories.head(10).to_dict()
    else:
        num_categories = 0
        top_categories = {}
        
    # Geographic Availability
    lat_col = config['lat_col']
    lon_col = config['lon_col']
    
    has_lat = False
    has_lon = False
    valid_coords_count = 0
    lat_min, lat_max, lon_min, lon_max = None, None, None, None
    
    if lat_col in df.columns and lon_col in df.columns:
        has_lat = True
        has_lon = True
        # Cast to numeric
        valid_lats = pd.to_numeric(df[lat_col], errors='coerce')
        valid_lons = pd.to_numeric(df[lon_col], errors='coerce')
        
        # Valid non-zero coordinates
        valid_mask = valid_lats.notnull() & valid_lons.notnull() & (valid_lats != 0) & (valid_lons != 0)
        valid_coords_count = valid_mask.sum()
        
        if valid_coords_count > 0:
            lat_min = float(valid_lats[valid_mask].min())
            lat_max = float(valid_lats[valid_mask].max())
            lon_min = float(valid_lons[valid_mask].min())
            lon_max = float(valid_lons[valid_mask].max())
            
    print(f"Records: {num_records:,}")
    print(f"Columns ({num_cols}): {list(df.columns)}")
    print(f"Date Range: {date_range_str}")
    print(f"Unique Categories: {num_categories}")
    print(f"Latitude Available: {has_lat} (Valid: {valid_coords_count:,})")
    print(f"Longitude Available: {has_lon} (Valid: {valid_coords_count:,})")
    print(f"Missing Values: {total_missing:,}")
    print(f"Duplicate IDs: {duplicate_ids:,}")
    if valid_coords_count > 0:
        print(f"Bounding Box: Lat [{lat_min:.4f}, {lat_max:.4f}], Lon [{lon_min:.4f}, {lon_max:.4f}]")
        
    summary_dict = {
        'Dataset': name,
        'Records': num_records,
        'Columns': num_cols,
        'Date Range': date_range_str,
        'Crime Categories': num_categories,
        'Latitude Available': f"Yes ({valid_coords_count:,} valid)" if has_lat else "No",
        'Longitude Available': f"Yes ({valid_coords_count:,} valid)" if has_lon else "No",
        'Missing Values': int(total_missing)
    }
    
    detailed_info = {
        'name': name,
        'records': num_records,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': null_counts.to_dict(),
        'duplicate_ids': int(duplicate_ids),
        'date_range': date_range_str,
        'num_categories': num_categories,
        'top_categories': top_categories,
        'category_distribution': categories.to_dict() if cat_col in df.columns else {},
        'has_lat': has_lat,
        'has_lon': has_lon,
        'valid_coords_count': int(valid_coords_count),
        'bounding_box': {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        } if valid_coords_count > 0 else None
    }
    
    return summary_dict, detailed_info

def main():
    os.makedirs('results/reports', exist_ok=True)
    summaries = []
    details = {}
    
    for name, config in DATASETS.items():
        summary, detail = inspect_dataset(name, config)
        summaries.append(summary)
        details[name] = detail
        
    summary_df = pd.DataFrame(summaries)
    summary_csv_path = 'results/reports/dataset_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary table to {summary_csv_path}")
    print(summary_df.to_markdown())
    
    with open('results/reports/detailed_inspection_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(details, f, indent=2)
    print("Detailed metrics saved to results/reports/detailed_inspection_metrics.json")

if __name__ == '__main__':
    main()
