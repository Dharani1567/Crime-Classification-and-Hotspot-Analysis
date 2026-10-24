"""
components/metric_cards.py
Renders the top four metric cards matching the reference image layout and subtle accent badges.
"""

import streamlit as st

def render_metric_cards(df, cfg):
    if df is None or df.empty:
        total_records = "Not Available"
        crime_categories = "Not Available"
        unique_locations = "Not Available"
        date_range = "Not Available"
    else:
        total_records = f"{len(df):,}"
        cat_col = cfg['cat_col']
        crime_categories = f"{df[cat_col].nunique():,}" if cat_col in df.columns else "Not Available"
        
        loc_col = cfg['location_col']
        if loc_col in df.columns:
            unique_locations = f"{df[loc_col].nunique():,}"
        elif 'spatial_cell_id' in df.columns:
            unique_locations = f"{df['spatial_cell_id'].nunique():,}"
        else:
            unique_locations = "Not Available"
            
        date_col = cfg['date_col']
        if date_col in df.columns and not df[date_col].isnull().all():
            min_date = df[date_col].min().strftime('%Y-%m')
            max_date = df[date_col].max().strftime('%Y-%m')
            date_range = f"{min_date} to {max_date}"
        else:
            date_range = "Not Available"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card card-blue">
            <div class="metric-icon" style="background: rgba(30, 144, 255, 0.15); color: #1e90ff;">
                📊
            </div>
            <div>
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{total_records}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card card-green">
            <div class="metric-icon" style="background: rgba(46, 204, 113, 0.15); color: #2ecc71;">
                ⚠️
            </div>
            <div>
                <div class="metric-label">Crime Categories</div>
                <div class="metric-value">{crime_categories}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card card-red">
            <div class="metric-icon" style="background: rgba(231, 76, 60, 0.15); color: #e74c3c;">
                📍
            </div>
            <div>
                <div class="metric-label">Unique Locations</div>
                <div class="metric-value">{unique_locations}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card card-purple">
            <div class="metric-icon" style="background: rgba(155, 89, 182, 0.15); color: #9b59b6;">
                📅
            </div>
            <div>
                <div class="metric-label">Date Range</div>
                <div class="metric-value" style="font-size: 1.15rem;">{date_range}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
