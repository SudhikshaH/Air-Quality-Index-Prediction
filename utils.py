import shap
import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    print("Gemini API configured successfully!")
except Exception as e:
    st.error("Gemini API key not found! Add it to .streamlit/secrets.toml")
    raise e

GEMINI_MODEL = "gemini-2.5-flash"  

def interpret_shap(model, features, data_point):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_point)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim > 1:
        shap_values = shap_values.flatten()

    shap_df = pd.DataFrame({
        'feature': features,
        'shap_value': np.round(shap_values, 4)
    })
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values('abs_shap', ascending=False)

    pos = shap_df[shap_df['shap_value'] > 0].head(3)
    neg = shap_df[shap_df['shap_value'] < 0].head(3)

    return {
        'positive': dict(zip(pos['feature'], pos['shap_value'])),
        'negative': dict(zip(neg['feature'], neg['shap_value'])),
        'all_shap': dict(zip(shap_df['feature'], shap_df['shap_value']))
    }

def get_llm_explanation(aqi_value, shap_dict):
    pos_str = ", ".join([f"{k} (+{v:.1f})" for k, v in shap_dict['positive'].items()]) or "none"
    neg_str = ", ".join([f"{k} ({v:.1f})" for k, v in shap_dict['negative'].items()]) or "none"

    category = "Good" if aqi_value <= 50 else \
               "Satisfactory" if aqi_value <= 100 else \
               "Moderate" if aqi_value <= 200 else \
               "Poor" if aqi_value <= 300 else \
               "Very Poor" if aqi_value <= 400 else "Severe"

    prompt = f"""Current AQI is {aqi_value:.0f} ({category} air quality).
Main pollutants increasing AQI: {pos_str}
Factors helping improve it: {neg_str}
Write a short explanation in simple English
give 1-2 points on heath impacts 
give 1-2 points on  health tips in bullet points."""

    with st.spinner("Getting smart explanation from Google Gemini..."):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                    top_p=0.95
                )
            )
            return response.text.strip()
            
        except Exception as e:
            st.error("Gemini API error. Check your key or internet.")
            print(f"Gemini error: {e}")
            return f"""The air quality is {category.lower()} (AQI {aqi_value:.0f}).
High pollution levels may affect sensitive groups.
• Limit outdoor time
• Keep windows closed
• Use air purifier if possible"""