"""
src/hotspot_classification.py
Part 4: Spatial Crime Hotspot Classification Across Four Regions

Task:
- Classify geographic spatial units into 3 concentration tiers:
    1. Low Crime Concentration
    2. Medium Crime Concentration
    3. High Crime Concentration
  using documented quantile-based thresholds (tertiles: 33.3%, 66.7%).
- Because population data is unavailable at the fine grid-cell level in raw feeds,
  the target represents incident concentration rather than population-adjusted risk.

Features:
- Prior 7-day total crime count
- Prior 30-day total crime count
- Prior 7-day theft count
- Prior 7-day assault / violent crime count
- Prior period crime count
- Geographic grid cell coords / administrative units
- Temporal features (hour, day of week, month, cyclical encodings)
- Strict Leakage Prevention: Calculated strictly on past records using pandas rolling with closed='left'

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

Spatial Analysis:
- DBSCAN exploratory clustering on valid geographic coordinates
- Geographic hotspot maps showing spatial concentration tiers and DBSCAN clusters

Evaluation:
- Chronological train/test split (80% train, 20% test)
- Macro-F1 (primary), Accuracy, Weighted-F1, Macro Precision, Macro Recall, Confusion Matrix
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

REGION_CONFIGS = {
    'chicago': {
        'name': 'Chicago',
        'file': 'data/processed/chicago/chicago_processed.csv',
        'geo_col': 'district',
        'grid_size': '0.01 degree (~1.1 km) & Police District',
        'theft_patterns': ['THEFT', 'BURGLARY', 'ROBBERY', 'MOTOR VEHICLE THEFT'],
        'assault_patterns': ['BATTERY', 'ASSAULT'],
        'cat_features': ['district', 'community_area'],
        'num_features': ['hour', 'sin_hour', 'cos_hour', 'day_of_week_num', 'sin_dayofweek', 'cos_dayofweek', 'is_weekend', 'month', 'latitude', 'longitude']
    },
    'los_angeles': {
        'name': 'Los Angeles',
        'file': 'data/processed/los_angeles/los_angeles_processed.csv',
        'geo_col': 'area_name',
        'grid_size': '0.01 degree (~1.1 km) & LAPD Division',
        'theft_patterns': ['THEFT', 'BURGLARY', 'ROBBERY', 'STOLEN'],
        'assault_patterns': ['BATTERY', 'ASSAULT'],
        'cat_features': ['area_name', 'area'],
        'num_features': ['hour', 'sin_hour', 'cos_hour', 'day_of_week_num', 'sin_dayofweek', 'cos_dayofweek', 'is_weekend', 'month', 'latitude', 'longitude']
    },
    'san_francisco': {
        'name': 'San Francisco',
        'file': 'data/processed/san_francisco/san_francisco_processed.csv',
        'geo_col': 'police_district',
        'grid_size': '0.01 degree (~1.1 km) & Police District',
        'theft_patterns': ['THEFT', 'BURGLARY', 'ROBBERY'],
        'assault_patterns': ['ASSAULT'],
        'cat_features': ['police_district', 'analysis_neighborhood'],
        'num_features': ['hour', 'sin_hour', 'cos_hour', 'day_of_week_num', 'sin_dayofweek', 'cos_dayofweek', 'is_weekend', 'month', 'latitude', 'longitude', 'has_valid_coordinates']
    },
    'uk': {
        'name': 'UK',
        'file': 'data/processed/uk/uk_processed.csv',
        'geo_col': 'street_name',
        'grid_size': '0.02 degree (~2.2 km) grid cell',
        'theft_patterns': ['theft', 'robbery', 'burglary', 'shoplifting'],
        'assault_patterns': ['violent-crime', 'public-order'],
        'cat_features': ['street_name', 'location_type'],
        'num_features': ['month_num', 'sin_month', 'cos_month', 'latitude', 'longitude']
    }
}

def build_hotspot_features(region_key, config, sample_size=40000):
    print(f"\n==================================================")
    print(f"ENGINEERING HOTSPOT FEATURES: {config['name'].upper()}")
    print(f"==================================================")
    
    df = pd.read_csv(config['file'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    df = df.sort_values('datetime').reset_index(drop=True)

    # 1. Geographic Grid Discretization
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['grid_lat'] = np.floor(df['latitude'] / 0.01) * 0.01
        df['grid_lon'] = np.floor(df['longitude'] / 0.01) * 0.01
        df['spatial_cell_id'] = (
            df[config['geo_col']].astype(str) + "_" +
            df['grid_lat'].round(2).astype(str) + "_" +
            df['grid_lon'].round(2).astype(str)
        )
    else:
        df['spatial_cell_id'] = df[config['geo_col']].astype(str)

    # Subsample if dataset is large, preserving chronological order
    if len(df) > sample_size:
        step = len(df) // sample_size
        df = df.iloc[::step].head(sample_size).reset_index(drop=True)

    # 2. Crime Type Masks for past historical indicator
    theft_regex = '|'.join(config['theft_patterns'])
    assault_regex = '|'.join(config['assault_patterns'])
    df['is_theft'] = df['crime_category'].astype(str).str.contains(theft_regex, case=False, na=False).astype(int)
    df['is_assault'] = df['crime_category'].astype(str).str.contains(assault_regex, case=False, na=False).astype(int)
    df['is_crime'] = 1

    # 3. Leakage-Free Rolling Aggregations per Geographic Area
    print(f"Computing leakage-free historical features for {config['name']} (closed='left')...")
    df['temp_date'] = df['datetime']
    df_indexed = df.set_index('temp_date')

    # Rolling window sizing
    is_monthly = (region_key == 'uk')
    w_7d = '60D' if is_monthly else '7D'
    w_30d = '90D' if is_monthly else '30D'
    w_prev = '14D' if is_monthly else '1D'

    geo_group = df_indexed.groupby(config['geo_col'])
    
    # 7-day total crime
    roll_7d = geo_group['is_crime'].rolling(w_7d, closed='left').sum().reset_index(level=0, drop=True)
    # 30-day total crime
    roll_30d = geo_group['is_crime'].rolling(w_30d, closed='left').sum().reset_index(level=0, drop=True)
    # 7-day theft
    roll_theft_7d = geo_group['is_theft'].rolling(w_7d, closed='left').sum().reset_index(level=0, drop=True)
    # 7-day assault
    roll_assault_7d = geo_group['is_assault'].rolling(w_7d, closed='left').sum().reset_index(level=0, drop=True)
    # Immediate prior period
    roll_prev = geo_group['is_crime'].rolling(w_prev, closed='left').sum().reset_index(level=0, drop=True)

    df['prev_7d_crime_count'] = roll_7d.fillna(0).values
    df['prev_30d_crime_count'] = roll_30d.fillna(0).values
    df['prev_7d_theft_count'] = roll_theft_7d.fillna(0).values
    df['prev_7d_assault_count'] = roll_assault_7d.fillna(0).values
    df['prev_period_crime_count'] = roll_prev.fillna(0).values

    # 4. Target Construction: Quantile-based 3 Tiers (Low, Medium, High Concentration)
    # Using 33.3% and 66.7% tertiles of recent activity
    # Note: Population data is unavailable at the fine grid-cell level; target represents incident concentration.
    q33 = df['prev_7d_crime_count'].quantile(0.333)
    q67 = df['prev_7d_crime_count'].quantile(0.667)
    print(f"Quantile thresholds for {config['name']}: 33rd={q33:.1f}, 67th={q67:.1f}")

    def assign_tier(val):
        if val <= q33:
            return 0 # Low Crime Concentration
        elif val <= q67:
            return 1 # Medium Crime Concentration
        else:
            return 2 # High Crime Concentration

    df['hotspot_tier'] = df['prev_7d_crime_count'].apply(assign_tier)
    tier_dist = df['hotspot_tier'].value_counts(normalize=True).to_dict()
    print(f"Hotspot Tier Distribution: Low={tier_dist.get(0, 0):.2%}, Med={tier_dist.get(1, 0):.2%}, High={tier_dist.get(2, 0):.2%}")

    # Temporal continuous cyclical features
    if 'hour' in df.columns:
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['sin_dayofweek'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7.0)
        df['cos_dayofweek'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7.0)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    else:
        df['sin_month'] = np.sin(2 * np.pi * df['month_num'] / 12.0)
        df['cos_month'] = np.cos(2 * np.pi * df['month_num'] / 12.0)

    # 5. Chronological Train/Test Split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    y_train = train_df['hotspot_tier'].values
    y_test = test_df['hotspot_tier'].values

    # Feature List
    hist_features = [
        'prev_30d_crime_count',
        'prev_7d_theft_count',
        'prev_7d_assault_count',
        'prev_period_crime_count'
    ]
    num_cols = config['num_features'] + hist_features

    # Standard scaling strictly on train
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[num_cols].fillna(0).values)
    X_test_num = scaler.transform(test_df[num_cols].fillna(0).values)

    # Frequency encode categoricals strictly on train
    X_train_cat_list, X_test_cat_list, cat_names = [], [], []
    for c in config['cat_features']:
        fmap = train_df[c].astype(str).value_counts(normalize=True).to_dict()
        X_train_cat_list.append(train_df[c].astype(str).map(fmap).fillna(0).values.reshape(-1, 1))
        X_test_cat_list.append(test_df[c].astype(str).map(fmap).fillna(0).values.reshape(-1, 1))
        cat_names.append(f"{c}_freq")

    X_train = np.hstack([X_train_num] + X_train_cat_list)
    X_test = np.hstack([X_test_num] + X_test_cat_list)
    feature_names = num_cols + cat_names

    return df, X_train, X_test, y_train, y_test, feature_names

def run_dbscan_clustering(region_key, df, config):
    """
    Applies DBSCAN exploratory clustering on geographic coordinates.
    Note: DBSCAN clusters represent spatial density anomalies and are not
    automatically treated as statistically significant hotspots.
    """
    print(f"\n>> Running DBSCAN Exploratory Spatial Clustering: {config['name']}...")
    valid_coords = df[df['latitude'].notnull() & df['longitude'].notnull() & (df['latitude'] != 0) & (df['longitude'] != 0)]
    
    if len(valid_coords) > 10000:
        valid_coords = valid_coords.sample(10000, random_state=42)

    # Convert coordinates to radians for haversine metric
    coords_rad = np.radians(valid_coords[['latitude', 'longitude']].values)
    
    # 1 km in radians: 1 / 6371.0088
    eps_rad = 1.0 / 6371.0088
    db = DBSCAN(eps=eps_rad, min_samples=30, metric='haversine')
    clusters = db.fit_predict(coords_rad)
    valid_coords = valid_coords.copy()
    valid_coords['dbscan_cluster'] = clusters

    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = (clusters == -1).sum()
    print(f"   DBSCAN identified {n_clusters} spatial clusters ({n_noise} noise points).")

    # Plot DBSCAN & Hotspot visualization
    figure_dir = f"results/figures/hotspot_classification/{region_key}"
    os.makedirs(figure_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: DBSCAN Clusters
    scatter1 = axes[0].scatter(
        valid_coords['longitude'], valid_coords['latitude'],
        c=valid_coords['dbscan_cluster'], cmap='tab20', s=8, alpha=0.6
    )
    axes[0].set_title(f"DBSCAN Exploratory Spatial Clusters\n({config['name']}: {n_clusters} clusters)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    # Right: 3-Tier Hotspot Concentration
    tier_colors = {0: '#2b83ba', 1: '#ffffbf', 2: '#d7191c'}
    tier_names = {0: 'Low', 1: 'Medium', 2: 'High'}
    for t in [0, 1, 2]:
        sub = valid_coords[valid_coords['hotspot_tier'] == t]
        axes[1].scatter(
            sub['longitude'], sub['latitude'],
            c=tier_colors[t], label=f"{tier_names[t]} Concentration",
            s=8, alpha=0.5
        )
    axes[1].set_title(f"3-Tier Crime Concentration Hotspots\n({config['name']})")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].legend(loc='lower left')

    plt.tight_layout()
    map_path = f"{figure_dir}/spatial_hotspot_visualization.png"
    plt.savefig(map_path, dpi=200)
    plt.close()
    print(f"   Saved spatial visualization to {map_path}")

    return n_clusters, n_noise

def train_hotspot_models(region_key, config):
    df, X_train, X_test, y_train, y_test, feature_names = build_hotspot_features(region_key, config, sample_size=40000)
    tier_labels = ['Low Concentration', 'Medium Concentration', 'High Concentration']

    # Exploratory DBSCAN
    run_dbscan_clustering(region_key, df, config)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=500,
            C=1.0,
            solver='lbfgs',
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_split=15,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
    }

    model_dir = f"models/hotspot_classification/{region_key}"
    metric_dir = f"results/metrics/hotspot_classification/{region_key}"
    figure_dir = f"results/figures/hotspot_classification/{region_key}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    metrics_list = []

    for name, model in models.items():
        print(f"\n>> Training {name} for Hotspot Classification ({config['name']})...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"   Accuracy:    {acc:.4f}")
        print(f"   Macro Prec:  {macro_prec:.4f}")
        print(f"   Macro Rec:   {macro_rec:.4f}")
        print(f"   Macro-F1:    {macro_f1:.4f} (Primary)")
        print(f"   Weighted-F1: {weighted_f1:.4f}")

        safe_name = name.lower().replace(' ', '_')
        joblib.dump(model, f"{model_dir}/{safe_name}.joblib")

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=tier_labels, yticklabels=tier_labels)
        plt.title(f"Hotspot Confusion Matrix: {name} ({config['name']})\nMacro-F1: {macro_f1:.4f}")
        plt.xlabel("Predicted Tier")
        plt.ylabel("Actual Tier")
        plt.tight_layout()
        plt.savefig(f"{figure_dir}/{safe_name}_cm.png", dpi=200)
        plt.close()

        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
            feat_df.to_csv(f"{metric_dir}/{safe_name}_feature_importance.csv", index=False)
            
            plt.figure(figsize=(9, 5))
            sns.barplot(data=feat_df.head(10), x='importance', y='feature', palette='mako')
            plt.title(f"Top 10 Hotspot Features: {name} ({config['name']})")
            plt.xlabel("Relative Importance")
            plt.tight_layout()
            plt.savefig(f"{figure_dir}/{safe_name}_feature_importance.png", dpi=200)
            plt.close()

        metrics_list.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Macro Precision': round(macro_prec, 4),
            'Macro Recall': round(macro_rec, 4),
            'Macro-F1': round(macro_f1, 4),
            'Weighted-F1': round(weighted_f1, 4)
        })

    results_df = pd.DataFrame(metrics_list)
    results_csv = f"{metric_dir}/model_comparison.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved Hotspot Comparison Table to {results_csv}")
    print(results_df.to_markdown(index=False))

    # Model comparison plot
    plt.figure(figsize=(8, 5))
    df_plot = results_df.melt(id_vars='Model', value_vars=['Macro-F1', 'Weighted-F1', 'Accuracy'], var_name='Metric', value_name='Score')
    sns.barplot(data=df_plot, x='Model', y='Score', hue='Metric', palette='magma')
    plt.title(f"Model Comparison: {config['name']} Hotspot Classification")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/model_comparison_plot.png", dpi=200)
    plt.close()

    return results_df

def run_all_hotspot_tasks():
    all_hotspot_results = {}
    for region_key, config in REGION_CONFIGS.items():
        res = train_hotspot_models(region_key, config)
        all_hotspot_results[config['name']] = res.to_dict(orient='records')

    with open('results/metrics/hotspot_classification/all_regions_summary.json', 'w', encoding='utf-8') as f:
        json.dump(all_hotspot_results, f, indent=2)
    print("\nALL FOUR REGIONS TRAINED AND EVALUATED FOR TASK 2 (HOTSPOT CLASSIFICATION).")

if __name__ == '__main__':
    run_all_hotspot_tasks()
