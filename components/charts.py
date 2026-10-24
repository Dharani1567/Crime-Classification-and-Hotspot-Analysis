"""
components/charts.py
Interactive Plotly charts replicating the overview and analytical panels of the reference image.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

COLOR_PALETTE = ['#2980b9', '#3498db', '#1abc9c', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#34495e', '#16a085', '#d35400']

def plot_crime_distribution(df, cfg, top_n=10):
    cat_col = cfg['cat_col']
    if cat_col not in df.columns:
        return None
        
    counts = df[cat_col].value_counts().head(top_n).reset_index()
    counts.columns = ['Category', 'Count']
    # Shorten names for clean display
    counts['Short_Category'] = counts['Category'].apply(lambda x: str(x)[:22] + '...' if len(str(x)) > 25 else str(x))
    
    fig = px.bar(
        counts,
        x='Short_Category',
        y='Count',
        color='Short_Category',
        color_discrete_sequence=COLOR_PALETTE,
        text='Count',
        hover_data={'Category': True, 'Count': ':,', 'Short_Category': False}
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(
        title={'text': "Crime Distribution by Category", 'font': {'size': 15, 'weight': 'bold', 'color': '#1e293b'}},
        xaxis_title=None,
        yaxis_title="Incident Count",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=30, r=20, t=45, b=65),
        xaxis_tickangle=-40,
        height=330
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    return fig

def plot_crime_trend(df, cfg):
    date_col = cfg['date_col']
    if date_col not in df.columns or df[date_col].isnull().all():
        return None

    # Group by Month or Week
    df_trend = df.set_index(date_col).resample('W').size().reset_index()
    df_trend.columns = ['Date', 'Incidents']
    if len(df_trend) < 4:
        df_trend = df.set_index(date_col).resample('M').size().reset_index()
        df_trend.columns = ['Date', 'Incidents']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trend['Date'],
        y=df_trend['Incidents'],
        mode='lines+markers',
        line=dict(color='#2980b9', width=2.5),
        marker=dict(size=5, color='#1e293b'),
        fill='tozeroy',
        fillcolor='rgba(41, 128, 185, 0.08)',
        hovertemplate='%{x|%b %Y}: %{y:,} crimes<extra></extra>'
    ))
    fig.update_layout(
        title={'text': "Crime Trend Over Time", 'font': {'size': 15, 'weight': 'bold', 'color': '#1e293b'}},
        xaxis_title=None,
        yaxis_title="Number of Incidents",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=30, r=20, t=45, b=30),
        height=330
    )
    fig.update_xaxes(showgrid=True, gridcolor='#f1f5f9')
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    return fig

def plot_model_comparison(cat_df, hot_df):
    if cat_df is None or hot_df is None:
        return None

    merged_data = []
    for _, row in cat_df.iterrows():
        merged_data.append({
            'Model': row['Model'],
            'Task': 'Crime Category',
            'Macro-F1': row['Macro-F1']
        })
    for _, row in hot_df.iterrows():
        merged_data.append({
            'Model': row['Model'],
            'Task': 'Hotspot Classification',
            'Macro-F1': row['Macro-F1']
        })
        
    comp_df = pd.DataFrame(merged_data)
    fig = px.bar(
        comp_df,
        x='Model',
        y='Macro-F1',
        color='Task',
        barmode='group',
        color_discrete_map={'Crime Category': '#3498db', 'Hotspot Classification': '#e67e22'}
    )
    fig.update_layout(
        title={'text': "Model Performance Comparison (Macro-F1)", 'font': {'size': 15, 'weight': 'bold', 'color': '#1e293b'}},
        yaxis_title="Macro F1-Score",
        xaxis_title=None,
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=30, r=20, t=55, b=40),
        height=330
    )
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9')
    return fig

def plot_incident_map(map_df, cfg):
    lat_col = cfg['lat_col']
    lon_col = cfg['lon_col']
    cat_col = cfg['cat_col']

    top_cats = map_df[cat_col].value_counts().head(7).index.tolist()
    map_df_plot = map_df.copy()
    map_df_plot['Display_Category'] = map_df_plot[cat_col].apply(lambda x: x if x in top_cats else 'Other')

    fig = px.scatter_mapbox(
        map_df_plot,
        lat=lat_col,
        lon=lon_col,
        color='Display_Category',
        zoom=cfg.get('default_zoom', 10),
        center=dict(lat=cfg['center'][0], lon=cfg['center'][1]),
        mapbox_style="carto-positron",
        hover_data={'Display_Category': True, lat_col: ':.4f', lon_col: ':.4f'},
        color_discrete_sequence=['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c', '#95a5a6'],
        title="Crime Incidents Map"
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        title={'text': "Crime Incidents Map (Sampled)", 'font': {'size': 15, 'weight': 'bold', 'color': '#1e293b'}},
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.85)"),
        height=330
    )
    return fig

def plot_hotspot_map(hotspot_df, cfg):
    if hotspot_df is None or hotspot_df.empty:
        return None

    # Map target tier: 0=Low, 1=Medium, 2=High
    hotspot_df_plot = hotspot_df.copy()
    tier_map = {0: 'Low Crime Area', 1: 'Medium Crime Area', 2: 'High Crime Area'}
    hotspot_df_plot['Hotspot_Tier'] = hotspot_df_plot['is_crime_hotspot'].map({0: 'Low Crime Area', 1: 'High Crime Area'})
    if 'hotspot_tier' in hotspot_df_plot.columns:
        hotspot_df_plot['Hotspot_Tier'] = hotspot_df_plot['hotspot_tier'].map(tier_map)

    fig = px.scatter_mapbox(
        hotspot_df_plot,
        lat='latitude',
        lon='longitude',
        color='Hotspot_Tier',
        zoom=cfg.get('default_zoom', 10),
        center=dict(lat=cfg['center'][0], lon=cfg['center'][1]),
        mapbox_style="carto-positron",
        color_discrete_map={
            'High Crime Area': '#e74c3c',
            'Medium Crime Area': '#f39c12',
            'Low Crime Area': '#2ecc71'
        },
        title="Hotspot Classification Map"
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    fig.update_layout(
        title={'text': "Hotspot Classification Map (Predicted Concentration)", 'font': {'size': 15, 'weight': 'bold', 'color': '#1e293b'}},
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.85)"),
        height=330
    )
    return fig

def plot_dbscan_map(dbscan_labels_df, cfg):
    if dbscan_labels_df is None or dbscan_labels_df.empty:
        return None

    df_plot = dbscan_labels_df.copy()
    df_plot['Cluster_Display'] = df_plot['cluster_label'].apply(
        lambda x: f"Cluster {x}" if x != -1 and x < 5 else ("Other Clusters" if x != -1 else "Noise")
    )
    
    # Custom color palette for DBSCAN
    colors = {
        'Cluster 0': '#e74c3c',
        'Cluster 1': '#3498db',
        'Cluster 2': '#2ecc71',
        'Cluster 3': '#9b59b6',
        'Cluster 4': '#e67e22',
        'Other Clusters': '#1abc9c',
        'Noise': '#95a5a6'
    }

    fig = px.scatter_mapbox(
        df_plot,
        lat=cfg['lat_col'],
        lon=cfg['lon_col'],
        color='Cluster_Display',
        zoom=cfg.get('default_zoom', 10),
        center=dict(lat=cfg['center'][0], lon=cfg['center'][1]),
        mapbox_style="carto-positron",
        color_discrete_map=colors,
        title="DBSCAN Clustering Map"
    )
    fig.update_traces(marker=dict(size=5, opacity=0.75))
    fig.update_layout(
        title={'text': "DBSCAN Clustering Map (Spatial Density)", 'font': {'size': 15, 'weight': 'bold', 'color': '#1e293b'}},
        margin=dict(l=0, r=0, t=35, b=0),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98, bgcolor="rgba(255,255,255,0.85)"),
        height=330
    )
    return fig
