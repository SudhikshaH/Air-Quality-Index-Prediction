import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
import kagglehub

def train_and_save_model():
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("rohanrao/air-quality-data-in-india")
    csv_path = os.path.join(path, "city_day.csv")
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=['Date'], dayfirst=False)
    print(f"Dataset shape: {df.shape}")
    # Focus on major pollutants and AQI
    df = df[['City', 'Date', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI', 'AQI_Bucket']]
    
    # Drop rows where AQI is missing
    df = df.dropna(subset=['AQI'])
    
    # Fill missing pollutants with median per city
    pollutant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    for col in pollutant_cols:
        df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    df[pollutant_cols] = df[pollutant_cols].fillna(df[pollutant_cols].median())

    # Feature Engineering
    df = df.sort_values(['City', 'Date'])

    # Lag features: last 1, 2, 7 days for PM2.5 and CO
    for lag in [1, 2, 7]:
        df[f'PM2.5_lag_{lag}'] = df.groupby('City')['PM2.5'].shift(lag)
        df[f'CO_lag_{lag}'] = df.groupby('City')['CO'].shift(lag)

    # Rolling mean (7-day)
    df['PM2.5_roll7'] = df.groupby('City')['PM2.5'].transform(lambda x: x.rolling(7, min_periods=1).mean())

    # Temporal features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Drop rows with NaN from lags
    df = df.dropna()

    # Select features
    feature_cols = pollutant_cols + [
        'PM2.5_lag_1', 'PM2.5_lag_2', 'PM2.5_lag_7',
        'CO_lag_1', 'CO_lag_2', 'CO_lag_7', 'PM2.5_roll7',
        'day_of_week', 'month', 'year', 'is_weekend'
    ]

    X = df[feature_cols]
    y = df['AQI']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror'
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.3f}")

    # Save model and feature list
    joblib.dump({
        'model': model,
        'features': feature_cols
    }, 'aqi_model.pkl')

    print("Model saved as 'aqi_model.pkl'")
    print(f"Using {len(feature_cols)} features.")

if __name__ == "__main__":
    train_and_save_model()