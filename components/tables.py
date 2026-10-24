"""
components/tables.py
Renders the bottom 3 summary cards matching the reference image:
1. Top 10 Crimes by Location
2. DBSCAN Cluster Summary
3. Hotspot Area Summary
"""

import streamlit as st
import pandas as pd

def render_summary_tables(df, cfg, dbscan_stats_df, hotspot_df):
    col1, col2, col3 = st.columns(3)

    # 1. Top 10 Crimes by Location
    with col1:
        st.markdown('<div class="table-container-title">Top 10 Crimes by Location</div>', unsafe_allow_html=True)
        loc_col = cfg['location_col']
        if loc_col in df.columns:
            top_locs = df[loc_col].value_counts().head(10).reset_index()
            top_locs.columns = ['Location', 'Count']
            top_locs['Rank'] = range(1, len(top_locs) + 1)
            # Reorder
            top_locs = top_locs[['Rank', 'Location', 'Count']]
            st.dataframe(
                top_locs,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Location": st.column_config.TextColumn("Location", width="medium"),
                    "Count": st.column_config.NumberColumn("Count", format="%d")
                },
                height=260
            )
        else:
            st.info("Location breakdown not available.")

    # 2. DBSCAN Cluster Summary
    with col2:
        st.markdown('<div class="table-container-title">DBSCAN Cluster Summary</div>', unsafe_allow_html=True)
        if dbscan_stats_df is not None and not dbscan_stats_df.empty:
            stats_show = dbscan_stats_df.head(10).copy()
            stats_show['Cluster'] = stats_show['cluster_id'].apply(lambda x: f"Cluster {x}")
            stats_show['Number of Points'] = stats_show['size']
            stats_show['Top Category'] = stats_show['top_categories'].apply(lambda x: str(x).split('(')[0].strip() if pd.notnull(x) else "Various")
            
            show_df = stats_show[['Cluster', 'Number of Points', 'Top Category']]
            st.dataframe(
                show_df,
                hide_index=True,
                column_config={
                    "Cluster": st.column_config.TextColumn("Cluster", width="small"),
                    "Number of Points": st.column_config.NumberColumn("Number of Points", format="%d"),
                    "Top Category": st.column_config.TextColumn("Top Category", width="medium")
                },
                height=260
            )
        else:
            st.info("DBSCAN cluster statistics not available.")

    # 3. Hotspot Area Summary
    with col3:
        st.markdown('<div class="table-container-title">Hotspot Area Summary</div>', unsafe_allow_html=True)
        area_col = cfg['area_col']
        if hotspot_df is not None and not hotspot_df.empty and area_col in hotspot_df.columns:
            area_summary = hotspot_df.groupby(area_col).agg(
                Crime_Count=('is_crime_hotspot', 'count'),
                Hotspot_Score=('is_crime_hotspot', 'mean')
            ).reset_index()
            
            def tier_label(score):
                if score > 0.65:
                    return 'High'
                elif score > 0.35:
                    return 'Medium'
                else:
                    return 'Low'

            area_summary['Predicted Class'] = area_summary['Hotspot_Score'].apply(tier_label)
            area_summary = area_summary.sort_values('Crime_Count', ascending=False).head(10)
            area_summary = area_summary[[area_col, 'Predicted Class', 'Crime_Count']]
            area_summary.columns = ['Area', 'Predicted Class', 'Crime Count']
            
            st.dataframe(
                area_summary,
                hide_index=True,
                column_config={
                    "Area": st.column_config.TextColumn("Area", width="medium"),
                    "Predicted Class": st.column_config.TextColumn("Predicted Class", width="small"),
                    "Crime Count": st.column_config.NumberColumn("Crime Count", format="%d")
                },
                height=260
            )
        else:
            st.info("Hotspot area summary not available.")
