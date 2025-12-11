import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
from utils import interpret_shap, get_llm_explanation
import shap
from matplotlib import pyplot as plt


st.set_page_config(page_title="AeroPred AI", layout="wide",)

# Load model
@st.cache_resource
def load_model():
    data = joblib.load("aqi_model.pkl")
    return data['model'], data['features']

model, features = load_model()

# Title
st.title("AeroPred AI : Air Quality Index Predictor")
st.markdown("Powered by Local LLM (Ollama)")

# Sidebar inputs
st.sidebar.header("Input Current Air Quality Data")

col1, col2 = st.sidebar.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 1000.0, 45.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, 1000.0, 80.0)
    no2 = st.number_input("NO₂ (µg/m³)", 0.0, 300.0, 35.0)
    co = st.number_input("CO (mg/m³)", 0.0, 20.0, 1.2)

with col2:
    so2 = st.number_input("SO₂ (µg/m³)", 0.0, 300.0, 15.0)
    o3 = st.number_input("O₃ (µg/m³)", 0.0, 400.0, 60.0)
    no = st.number_input("NO (µg/m³)", 0.0, 200.0, 20.0)
    nh3 = st.number_input("NH₃ (µg/m³)", 0.0, 100.0, 10.0, help="Ammonia")

# Simulate lag features (using user input as proxy)
lag_pm25 = st.sidebar.slider("Recent PM2.5 Trend (Last 7 Days Avg)", 10.0, 300.0, pm25 * 1.1)
lag_co = st.sidebar.slider("Recent CO Trend (Last 7 Days Avg)", 0.5, 10.0, co)

# Predict button
if st.sidebar.button("Predict AQI", type="primary", use_container_width=True):
    with st.spinner("Analyzing air quality..."):
        # Create input DataFrame
        input_data = {
            'PM2.5': pm25, 'PM10': pm10, 'NO': no, 'NO2': no2, 'NOx': no2*1.2,
            'NH3': nh3, 'CO': co, 'SO2': so2, 'O3': o3,
            'PM2.5_lag_1': pm25 * 0.95,
            'PM2.5_lag_2': pm25 * 0.9,
            'PM2.5_lag_7': lag_pm25,
            'CO_lag_1': co * 0.95,
            'CO_lag_2': co * 0.9,
            'CO_lag_7': lag_co,
            'PM2.5_roll7': lag_pm25,
            'day_of_week': datetime.now().weekday(),
            'month': datetime.now().month,
            'year': datetime.now().year,
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0
        }

        df_input = pd.DataFrame([input_data])[features]

        # Predict
        pred_aqi = model.predict(df_input)[0]

        # SHAP Explanation
        shap_dict = interpret_shap(model, features, df_input)

        # LLM Explanation
        llm_text = get_llm_explanation(pred_aqi, shap_dict)

        # AQI Category
        def get_aqi_bucket(aqi):
            if aqi <= 50: return "Good", "#00e400"
            elif aqi <= 100: return "Satisfactory", "#92d050"
            elif aqi <= 200: return "Moderate", "#ff0"
            elif aqi <= 300: return "Poor", "#ff7e00"
            elif aqi <= 400: return "Very Poor", "#ff0000"
            else: return "Severe", "#99004c"

        bucket, color = get_aqi_bucket(pred_aqi)

        # Display Results
        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            st.metric("Predicted AQI", f"{pred_aqi:.1f}", delta=None)
            st.markdown(f"### <span style='color:{color}'>{bucket}</span>", unsafe_allow_html=True)

        with col2:
            st.success("**AI Explanation**")
            st.write(llm_text)

        with col3:
            st.info("**Top Contributors**")
            for f, v in list(shap_dict['positive'].items())[:3]:
                st.markdown(f"🔴 **{f}** +{v}")
            for f, v in list(shap_dict['negative'].items())[:3]:
                st.markdown(f"🟢 **{f}** {v}")

        # SHAP Waterfall Plot
        st.markdown("### Feature Impact Breakdown (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_input)

        # Flatten correctly
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values
        if sv.ndim > 1:
            sv = sv.flatten()

        # Create explanation object
        explanation = shap.Explanation(
            values=sv,
            base_values=explainer.expected_value,
            data=df_input.iloc[0].values,
            feature_names=features
        )

        fig = plt.figure()
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig)
        plt.close()

        # Optional: Plotly bar
        contrib = pd.DataFrame({
            'Feature': features,
            'Impact': shap_values[0]
        }).sort_values('Impact', key=abs, ascending=False).head(10)

        fig2 = go.Figure(go.Bar(
            x=contrib['Impact'],
            y=contrib['Feature'],
            orientation='h',
            marker_color=['red' if x > 0 else 'green' for x in contrib['Impact']]
        ))
        fig2.update_layout(title="Top 10 Feature Contributions", height=400)
        st.plotly_chart(fig2, use_container_width=True)
