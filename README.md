# 🌫️ AeroPred AI: Air Quality Index Predictor

**A Multi-Layered, Explainable AI (XAI) and Generative AI Solution for AQI Forecasting.**

AeroPred AI is a public health-focused machine learning project that predicts the Air Quality Index (AQI) and provides transparent, natural-language explanations for its predictions. It combines a high-performance XGBoost model with the SHAP (SHapley Additive exPlanations) framework and a Large Language Model (LLM) for actionable, user-friendly advisories.

## ✨ Features

* **High-Accuracy Prediction:** Uses a trained XGBoost Regressor for daily AQI forecasting.
* **Explainable AI (XAI):** Implements **SHAP** to interpret the 'black-box' model, visually showing the contribution of each pollutant ($\text{PM2.5}$, $\text{CO}$, etc.) to the final AQI score.
* **Generative AI Interpretation:** Integrates a **Gemini API (gemini-2.5-flash)** to translate the complex numerical prediction and SHAP analysis into easy-to-understand health advisories and tips.
* **Streamlit Web App:** Deployed as an intuitive, interactive web application for easy public access.

## 🚀 Setup and Installation

### Prerequisites

You need Python 3.8+ and access to a terminal. Gemini API key generated.

### Dependencies Installation
1.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Running the Project

1.  **Run the Model Trainer:** This script downloads the dataset, trains the XGBoost model, and saves the serialized model file (`aqi_model.pkl`).
    ```bash
    python model_trainer.py
    ```
2.  **Start the Streamlit Application:** This will launch the web application in your browser.
    ```bash
    streamlit run app.py
    ```
## 💾 Dataset

The project relies on publicly available air quality data for Indian cities.

* **Dataset Source:** [Air Quality Data in India (2015 - 2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
* **File Used:** `city_day.csv`
* **Creator:** Rohan Rao


## 🛠️ Project Files

| File | Description |
| :--- | :--- |
| `app.py` | Main Streamlit application. Handles user input, model loading, prediction, and SHAP visualization. |
| `model_trainer.py` | Script for data cleaning, feature engineering, XGBoost training, and saving the model artifact. |
| `utils.py` | Utility functions, including LLM prompt construction, Gemini API communication, and LLM server management. |
| `requirements.txt` | List of Python dependencies (`streamlit`, `xgboost`, `shap`, etc.). |
| `aqi_model.pkl` | Serialized trained XGBoost model and feature list (created by `model_trainer.py`). |
| `city_day.csv` | The raw air quality dataset. |

***
