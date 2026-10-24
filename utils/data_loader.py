"""
utils/data_loader.py
High-performance cached data loaders for Streamlit dashboard.
Caches processed datasets, DBSCAN outputs, and model metrics.
"""

import os
import json
import joblib
import pandas as pd
import streamlit as st
from utils.dataset_config import DATASETS

@st.cache_data(show_spinner=False)
def load_processed_data(dataset_name, sample_for_map=5000):
    cfg = DATASETS.get(dataset_name)
    if not cfg or not os.path.exists(cfg['processed_file']):
        return None, None
    
    df = pd.read_csv(cfg['processed_file'])
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed')
    
    # Pre-sample map points for responsive rendering
    lat_col = cfg['lat_col']
    lon_col = cfg['lon_col']
    valid_map_df = df[df[lat_col].notnull() & df[lon_col].notnull() & (df[lat_col] != 0) & (df[lon_col] != 0)]
    if len(valid_map_df) > sample_for_map:
        map_df = valid_map_df.sample(sample_for_map, random_state=42).copy()
    else:
        map_df = valid_map_df.copy()
        
    return df, map_df

@st.cache_data(show_spinner=False)
def load_dbscan_data(dataset_name):
    cfg = DATASETS.get(dataset_name)
    if not cfg:
        return None, None
        
    labels_file = cfg['cluster_file']
    stats_file = cfg['cluster_stats_file']
    
    df_labels = pd.read_csv(labels_file) if os.path.exists(labels_file) else None
    df_stats = pd.read_csv(stats_file) if os.path.exists(stats_file) else None
    return df_labels, df_stats

@st.cache_data(show_spinner=False)
def load_hotspot_data(dataset_name, sample_size=5000):
    cfg = DATASETS.get(dataset_name)
    if not cfg or not os.path.exists(cfg['task2_file']):
        return None
        
    df = pd.read_csv(cfg['task2_file'])
    valid = df[df['latitude'].notnull() & df['longitude'].notnull() & (df['latitude'] != 0) & (df['longitude'] != 0)]
    if len(valid) > sample_size:
        return valid.sample(sample_size, random_state=42).copy()
    return valid.copy()

@st.cache_data(show_spinner=False)
def load_model_metrics(dataset_name):
    cfg = DATASETS.get(dataset_name)
    if not cfg:
        return None, None
        
    cat_df = pd.read_csv(cfg['category_metrics_file']) if os.path.exists(cfg['category_metrics_file']) else None
    hot_df = pd.read_csv(cfg['hotspot_metrics_file']) if os.path.exists(cfg['hotspot_metrics_file']) else None
    return cat_df, hot_df

@st.cache_data(show_spinner=False)
def load_all_model_comparison():
    csv_path = 'results/metrics/model_comparison.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def load_trained_model(models_dir, model_name_slug):
    # Try .pkl first, then .joblib
    pkl_path = os.path.join(models_dir, f"{model_name_slug}.pkl")
    joblib_path = os.path.join(models_dir, f"{model_name_slug}.joblib")
    
    if os.path.exists(pkl_path):
        return joblib.load(pkl_path)
    elif os.path.exists(joblib_path):
        return joblib.load(joblib_path)
    return None
