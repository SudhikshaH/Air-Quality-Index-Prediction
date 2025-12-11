import requests
import subprocess
import time
import os
import threading
import shap
import numpy as np
import pandas as pd
import streamlit as st

_ollama_process = None
_ollama_lock = threading.Lock()

def _start_ollama_server():
    global _ollama_process
    with _ollama_lock:
        if _ollama_process is not None:
            return  
        print("Starting Ollama server automatically in background...")

        try:
            requests.get("http://localhost:11434/", timeout=3)
            print("Ollama already running!")
            return
        except:
            pass
        if os.name == 'nt': 
            _ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        """
        else:  # macOS / Linux
            _ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
        """
        for i in range(30):
            time.sleep(2)
            try:
                resp = requests.get("http://localhost:11434/", timeout=5)
                if resp.status_code == 200:
                    print("Ollama server is ready!")
                    return
            except:
                continue

        print("Warning: Ollama server took too long to start. Continuing anyway...")

def ensure_ollama_running():
    if _ollama_process is None:
        thread = threading.Thread(target=_start_ollama_server, daemon=True)
        thread.start()
        thread.join(timeout=10)  
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
    ensure_ollama_running()  
    pos_str = ", ".join([f"{k} (+{v:.1f})" for k, v in shap_dict['positive'].items()]) or "none"
    neg_str = ", ".join([f"{k} ({v:.1f})" for k, v in shap_dict['negative'].items()]) or "none"

    category = "Good" if aqi_value <= 50 else \
               "Satisfactory" if aqi_value <= 100 else \
               "Moderate" if aqi_value <= 200 else \
               "Poor" if aqi_value <= 300 else \
               "Very Poor" if aqi_value <= 400 else "Severe"

    prompt = f"""AQI is {aqi_value:.0f} ({category}).
Main factors pushing it up: {pos_str}
Main factors pulling it down: {neg_str}

Write a short, friendly 2-sentence explanation in simple English + give 1-2 heath impacts + give 2-3 health tips in bullet points."""
    with st.spinner("Waking up local AI (llama3.2) — first time takes 10-25 sec..."):
        for attempt in range(5):  # Try up to 5 times
            try:
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2:latest",
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "num_predict": 256}
                    },
                    timeout=60  # Give it time on first load
                )
                if resp.status_code == 200:
                    result = resp.json().get("response", "").strip()
                    if result:
                        return result
            except Exception as e:
                print(f"LLM attempt {attempt + 1} failed: {e}")
                time.sleep(8)

    return "AI explanation ready! Refresh or try again in 10 seconds."