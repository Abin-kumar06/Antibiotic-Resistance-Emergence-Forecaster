import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

st.set_page_config(page_title="Antibiotic Resistance Forecaster", layout="wide", page_icon="🦠")

# Load CSS for premium styling
st.markdown("""
<style>
    .main {background-color: #0d1117; color: #c9d1d9;}
    h1, h2, h3 {color: #58a6ff;}
    .stMetric {background-color: #161b22; padding: 10px; border-radius: 8px; border: 1px solid #30363d;}
    div[data-testid="stSidebar"] {background-color: #161b22;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if os.path.exists('data/synthetic_antibiotic_data.csv'):
        return pd.read_csv('data/synthetic_antibiotic_data.csv', parse_dates=['Treatment_Date'])
    return pd.DataFrame()

@st.cache_data
def load_metrics():
    if os.path.exists('evaluation/metrics/results.json'):
        with open('evaluation/metrics/results.json', 'r') as f:
            return json.load(f)
    return {}

df = load_data()
metrics = load_metrics()

st.title("🦠 Antibiotic Resistance Emergence Forecaster")
st.markdown("Real-time monitoring and early warning system for localized antibiotic resistance outbreaks.")

if df.empty:
    st.warning("Data not found. Please run the data generator and preprocessing pipelines.")
else:
    # Sidebar
    st.sidebar.header("Filters")
    regions = df['Region'].unique()
    selected_region = st.sidebar.selectbox("Select Region", ["All"] + list(regions))
    
    # Filter data
    if selected_region != "All":
        filtered_df = df[df['Region'] == selected_region]
    else:
        filtered_df = df
        
    filtered_df['Resistant_Numeric'] = filtered_df['Resistance_Status'].apply(lambda x: 1 if x == 'Resistant' else 0)

    # Top level metrics
    col1, col2, col3, col4 = st.columns(4)
    total_cases = len(filtered_df)
    res_cases = filtered_df['Resistant_Numeric'].sum()
    res_rate = (res_cases / total_cases) * 100 if total_cases > 0 else 0
    
    col1.metric("Total Prescriptions", f"{total_cases:,}")
    col2.metric("Resistant Cases", f"{res_cases:,}")
    col3.metric("Overall Resistance Rate", f"{res_rate:.1f}%")
    
    if 'XGBoost' in metrics:
        col4.metric("Model ROC-AUC", f"{metrics['XGBoost']['ROC-AUC']:.3f}")

    st.markdown("---")
    
    # Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Resistance Rate over Time")
        # Ensure treatment date is rounded to month for clear plotting
        trend_df = filtered_df.copy()
        trend_df['Month_Year'] = trend_df['Treatment_Date'].dt.to_period('M').astype(str)
        monthly_res = trend_df.groupby('Month_Year')['Resistant_Numeric'].mean().reset_index()
        monthly_res['Resistant_Numeric'] *= 100 # percentage
        
        fig1 = px.line(monthly_res, x='Month_Year', y='Resistant_Numeric', markers=True, 
                       title="Monthly Resistance Emergence Rate (%)",
                       color_discrete_sequence=['#ff7b72'])
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.subheader("Resistance by Antibiotic & Strain")
        heatmap_data = pd.crosstab(filtered_df['Bacterial_Strain'], filtered_df['Antibiotic'], 
                                   values=filtered_df['Resistant_Numeric'], aggfunc='mean').fillna(0) * 100
        fig2 = px.imshow(heatmap_data, text_auto=".1f", aspect="auto", 
                         color_continuous_scale="Reds", title="Resistance % Heatmap")
        st.plotly_chart(fig2, use_container_width=True)

    # Alerts system
    st.subheader("🚨 Early Warning System Alerts")
    latest_month = filtered_df['Treatment_Date'].max() - pd.Timedelta(days=30)
    recent_df = filtered_df[filtered_df['Treatment_Date'] > latest_month]
    
    if len(recent_df) > 0:
        recent_res_rate = recent_df['Resistant_Numeric'].mean() * 100
        if recent_res_rate > 25:
            st.error(f"Critical Alert: Resistance rate in the last 30 days is unusually high at {recent_res_rate:.1f}%. Immediate review of stewardship programs is advised.")
        elif recent_res_rate > 15:
            st.warning(f"Warning: Resistance rate trending upwards ({recent_res_rate:.1f}%). Monitor specific regions and strains closely.")
        else:
            st.success(f"Status Normal: Resistance rates are stable ({recent_res_rate:.1f}%).")

    # SHAP feature importance
    st.subheader("Model Interpretability (XGBoost)")
    shap_path = 'evaluation/metrics/shap_summary.png'
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Feature Importance Plot", use_container_width=True)
    else:
        st.info("SHAP plot not yet generated. Please run evaluation pipeline.")

