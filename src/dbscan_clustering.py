"""
src/dbscan_clustering.py
Unsupervised Crime Spatial Clustering Using DBSCAN Across Four Regions

Task:
- Discover geographic clusters of crime incidents based on spatial density.
- Operates independently on:
  1. Chicago
  2. Los Angeles
  3. San Francisco
  4. UK (evaluated by major metropolitan sub-regions: London & Birmingham/North)
- Parameter Exploration & Sensitivity Analysis:
  * Tests combinations of eps (converted to Haversine radians on Earth radius R=6371 km)
    and min_samples.
- Evaluation:
  * Number of clusters, noise count and percentage, cluster size distributions.
  * Silhouette score (on sample of non-noise points when valid clusters > 1).
  * Crime category distributions within clusters.
  * Representative cluster centers (median lat/lon), geographic extents, and densities.
- Visualizations:
  * Incident maps with DBSCAN cluster assignments vs noise points.
  * Category distributions for top clusters.
  * Parameter sensitivity plots.
- Saves:
  * results/dbscan/<region>/cluster_labels.csv
  * results/dbscan/<region>/cluster_statistics.csv
  * results/dbscan/<region>/parameter_comparison.csv
  * Visual maps in results/dbscan/<region>/
  * Comprehensive report in results/reports/dbscan_analysis.md
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
EARTH_RADIUS_KM = 6371.0088

DATASET_CONFIGS = {
    'chicago': {
        'name': 'Chicago',
        'file': 'data/processed/chicago/chicago_processed.csv',
        'lat_col': 'latitude',
        'lon_col': 'longitude',
        'cat_col': 'crime_category',
        'param_grid': [
            {'eps_km': 0.3, 'min_samples': 25},
            {'eps_km': 0.5, 'min_samples': 30},  # Primary chosen configuration
            {'eps_km': 0.8, 'min_samples': 50},
            {'eps_km': 1.0, 'min_samples': 50}
        ],
        'primary_param': {'eps_km': 0.5, 'min_samples': 30}
    },
    'los_angeles': {
        'name': 'Los Angeles',
        'file': 'data/processed/los_angeles/los_angeles_processed.csv',
        'lat_col': 'latitude',
        'lon_col': 'longitude',
        'cat_col': 'crime_category',
        'param_grid': [
            {'eps_km': 0.4, 'min_samples': 30},
            {'eps_km': 0.6, 'min_samples': 40},  # Primary chosen configuration
            {'eps_km': 0.8, 'min_samples': 50},
            {'eps_km': 1.2, 'min_samples': 60}
        ],
        'primary_param': {'eps_km': 0.6, 'min_samples': 40}
    },
    'san_francisco': {
        'name': 'San Francisco',
        'file': 'data/processed/san_francisco/san_francisco_processed.csv',
        'lat_col': 'latitude',
        'lon_col': 'longitude',
        'cat_col': 'crime_category',
        'param_grid': [
            {'eps_km': 0.2, 'min_samples': 20},
            {'eps_km': 0.35, 'min_samples': 30}, # Primary chosen configuration
            {'eps_km': 0.5, 'min_samples': 40},
            {'eps_km': 0.8, 'min_samples': 50}
        ],
        'primary_param': {'eps_km': 0.35, 'min_samples': 30}
    },
    'uk': {
        'name': 'UK (Greater London Metro)',
        'file': 'data/processed/uk/uk_processed.csv',
        'lat_col': 'latitude',
        'lon_col': 'longitude',
        'cat_col': 'crime_category',
        'filter_box': {'lat_min': 51.35, 'lat_max': 51.65, 'lon_min': -0.25, 'lon_max': 0.05},
        'param_grid': [
            {'eps_km': 0.25, 'min_samples': 30},
            {'eps_km': 0.4, 'min_samples': 40},  # Primary chosen configuration
            {'eps_km': 0.6, 'min_samples': 50},
            {'eps_km': 0.8, 'min_samples': 60}
        ],
        'primary_param': {'eps_km': 0.4, 'min_samples': 40}
    }
}

def load_clean_geo_data(config, sample_size=15000):
    df = pd.read_csv(config['file'])
    lat_col = config['lat_col']
    lon_col = config['lon_col']
    
    # Filter valid non-null coordinates
    valid_mask = df[lat_col].notnull() & df[lon_col].notnull() & (df[lat_col] != 0) & (df[lon_col] != 0)
    df_valid = df[valid_mask].copy()

    # Optional regional filter (for UK city sub-clustering)
    if 'filter_box' in config:
        fb = config['filter_box']
        df_valid = df_valid[
            (df_valid[lat_col] >= fb['lat_min']) & (df_valid[lat_col] <= fb['lat_max']) &
            (df_valid[lon_col] >= fb['lon_min']) & (df_valid[lon_col] <= fb['lon_max'])
        ]

    if len(df_valid) > sample_size:
        df_valid = df_valid.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        df_valid = df_valid.reset_index(drop=True)

    print(f"Loaded {len(df_valid):,} valid coordinates for {config['name']}.")
    return df_valid

def run_parameter_sensitivity(df, config):
    lat_rad = np.radians(df[config['lat_col']].values)
    lon_rad = np.radians(df[config['lon_col']].values)
    coords_rad = np.column_stack([lat_rad, lon_rad])

    param_results = []
    print(f"\nEvaluating DBSCAN parameter sensitivity on {config['name']}...")

    for params in config['param_grid']:
        eps_km = params['eps_km']
        min_samples = params['min_samples']
        eps_rad = eps_km / EARTH_RADIUS_KM

        db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
        labels = db.fit_predict(coords_rad)

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        noise_pct = (n_noise / len(labels)) * 100

        # Calculate silhouette on non-noise points if valid
        sil_score = None
        non_noise_mask = (labels != -1)
        if n_clusters > 1 and non_noise_mask.sum() > n_clusters * 2:
            # Subsample for fast silhouette computation
            sample_mask = np.where(non_noise_mask)[0]
            if len(sample_mask) > 3000:
                sample_idx = np.random.choice(sample_mask, 3000, replace=False)
            else:
                sample_idx = sample_mask
            try:
                sil_score = float(silhouette_score(coords_rad[sample_idx], labels[sample_idx], metric='haversine'))
            except Exception:
                sil_score = None

        param_results.append({
            'eps_km': eps_km,
            'min_samples': min_samples,
            'clusters_found': n_clusters,
            'noise_points': n_noise,
            'noise_pct': round(noise_pct, 2),
            'silhouette_score': round(sil_score, 4) if sil_score is not None else 'N/A'
        })
        print(f"   eps={eps_km}km, min_samples={min_samples} -> Clusters: {n_clusters}, Noise: {noise_pct:.1f}%, Sil: {sil_score}")

    return pd.DataFrame(param_results)

def cluster_and_analyze(region_key, config):
    print(f"\n==================================================")
    print(f"DBSCAN SPATIAL CLUSTERING: {config['name'].upper()}")
    print(f"==================================================")
    
    out_dir = f"results/dbscan/{region_key}"
    os.makedirs(out_dir, exist_ok=True)
    
    df = load_clean_geo_data(config, sample_size=15000)
    lat_col = config['lat_col']
    lon_col = config['lon_col']
    cat_col = config['cat_col']

    # 1. Parameter Sensitivity Comparison
    param_df = run_parameter_sensitivity(df, config)
    param_csv = f"{out_dir}/parameter_comparison.csv"
    param_df.to_csv(param_csv, index=False)
    print(f"Saved parameter sensitivity table to {param_csv}")

    # Plot Parameter Sensitivity
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    x_labels = [f"eps={r['eps_km']}km\nms={r['min_samples']}" for _, r in param_df.iterrows()]
    x_pos = np.arange(len(x_labels))

    ax1.bar(x_pos - 0.15, param_df['clusters_found'], width=0.3, color='#3182bd', label='Clusters Found')
    ax2.plot(x_pos + 0.15, param_df['noise_pct'], color='#de2d26', marker='o', linewidth=2, label='Noise %')

    ax1.set_xlabel('Parameter Configuration')
    ax1.set_ylabel('Number of Clusters Found', color='#3182bd')
    ax2.set_ylabel('Noise Percentage (%)', color='#de2d26')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    plt.title(f"DBSCAN Parameter Sensitivity: {config['name']}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/parameter_sensitivity.png", dpi=200)
    plt.close()

    # 2. Fit Primary Clustering Model
    primary = config['primary_param']
    eps_rad = primary['eps_km'] / EARTH_RADIUS_KM
    min_samples = primary['min_samples']
    print(f"\nFitting Primary DBSCAN: eps={primary['eps_km']} km, min_samples={min_samples}...")

    coords_rad = np.column_stack([np.radians(df[lat_col].values), np.radians(df[lon_col].values)])
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
    labels = db.fit_predict(coords_rad)
    df['cluster_label'] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    noise_pct = (n_noise / len(labels)) * 100
    print(f"Primary Result: {n_clusters} clusters identified, {n_noise} noise points ({noise_pct:.2f}%).")

    # Save cluster labels
    label_df = df[['cluster_label', lat_col, lon_col, cat_col]].copy()
    label_df.to_csv(f"{out_dir}/cluster_labels.csv", index=False)

    # 3. Cluster Statistics & Metrics
    cluster_stats = []
    non_noise_df = df[df['cluster_label'] != -1]

    for cid in sorted(non_noise_df['cluster_label'].unique()):
        c_sub = non_noise_df[non_noise_df['cluster_label'] == cid]
        size = len(c_sub)
        med_lat = float(c_sub[lat_col].median())
        med_lon = float(c_sub[lon_col].median())
        lat_range = float(c_sub[lat_col].max() - c_sub[lat_col].min())
        lon_range = float(c_sub[lon_col].max() - c_sub[lon_col].min())
        
        # Approx bounding extent in km
        lat_km = lat_range * 111.0
        lon_km = lon_range * (111.0 * np.cos(np.radians(med_lat)))
        extent_km2 = max(0.01, lat_km * lon_km)
        density = round(size / extent_km2, 1)

        # Top 3 crime categories in this cluster
        top_cats = c_sub[cat_col].value_counts().head(3).to_dict()
        top_cats_str = ", ".join([f"{k} ({v})" for k, v in top_cats.items()])

        cluster_stats.append({
            'cluster_id': cid,
            'size': size,
            'pct_of_clustered_crimes': round((size / len(non_noise_df)) * 100, 2),
            'median_latitude': round(med_lat, 5),
            'median_longitude': round(med_lon, 5),
            'extent_approx_km2': round(extent_km2, 3),
            'estimated_density_per_km2': density,
            'top_categories': top_cats_str
        })

    stats_df = pd.DataFrame(cluster_stats)
    stats_csv = f"{out_dir}/cluster_statistics.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved cluster statistics to {stats_csv}")

    # 4. Geographic Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: All incidents with Cluster Color & Noise in Grey
    noise_sub = df[df['cluster_label'] == -1]
    axes[0].scatter(noise_sub[lon_col], noise_sub[lat_col], c='#bdbdbd', s=6, alpha=0.3, label=f'Noise ({n_noise})')
    
    # Plot non-noise clusters
    scatter = axes[0].scatter(
        non_noise_df[lon_col], non_noise_df[lat_col],
        c=non_noise_df['cluster_label'], cmap='tab20', s=12, alpha=0.7
    )
    axes[0].set_title(f"DBSCAN Spatial Clusters & Noise Points\n{config['name']} (eps={primary['eps_km']}km, min_samples={min_samples})")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].legend(loc='lower left')

    # Panel B: Cluster Size Barplot
    if not stats_df.empty:
        top_stats = stats_df.head(10)
        sns.barplot(data=top_stats, x='cluster_id', y='size', palette='crest', ax=axes[1])
        axes[1].set_title(f"Cluster Size Distribution (Top {len(top_stats)} Clusters)")
        axes[1].set_xlabel("Cluster ID (Arbitrary)")
        axes[1].set_ylabel("Incident Count")
        for i, v in enumerate(top_stats['size']):
            axes[1].text(i, v + max(top_stats['size']) * 0.01, str(v), ha='center', fontsize=8)

    plt.tight_layout()
    map_path = f"{out_dir}/dbscan_cluster_map.png"
    plt.savefig(map_path, dpi=200)
    plt.close()
    print(f"Saved spatial cluster map to {map_path}")

    # 5. Crime Category Distribution in Clusters Visualization
    if not stats_df.empty:
        top_cids = stats_df['cluster_id'].head(5).tolist()
        sub_top = df[df['cluster_label'].isin(top_cids)]
        top_overall_cats = sub_top[cat_col].value_counts().head(6).index.tolist()
        sub_top = sub_top[sub_top[cat_col].isin(top_overall_cats)]

        plt.figure(figsize=(10, 5))
        sns.countplot(data=sub_top, x='cluster_label', hue=cat_col, palette='tab10')
        plt.title(f"Crime Category Composition Across Top DBSCAN Clusters ({config['name']})")
        plt.xlabel("Cluster ID")
        plt.ylabel("Crime Count")
        plt.legend(title="Crime Category", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        cat_plot_path = f"{out_dir}/cluster_crime_category_composition.png"
        plt.savefig(cat_plot_path, dpi=200)
        plt.close()
        print(f"Saved category breakdown chart to {cat_plot_path}")

    return {
        'name': config['name'],
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'noise_pct': float(noise_pct),
        'primary_param': primary,
        'top_clusters': stats_df.head(5).to_dict(orient='records') if not stats_df.empty else []
    }

def run_all_dbscan():
    summary_results = {}
    for region_key, config in DATASET_CONFIGS.items():
        res = cluster_and_analyze(region_key, config)
        summary_results[config['name']] = res

    with open('results/dbscan/all_dbscan_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2)
    print("\nALL DBSCAN EXPERIMENTS COMPLETED.")

if __name__ == '__main__':
    run_all_dbscan()
