"""
Configuration file for AQI Prediction System
Store your API keys and configuration here
"""

# Hopsworks Configuration
HOPSWORKS_API_KEY = "8cgk2mCWFrd6GmH8.22VoIwMicv1OJxhnXUY97JA8wgq22iK92fRfp0nn3dW9vVJSbms90AHv7pmq9A4R"  # Get from hopsworks.ai
HOPSWORKS_PROJECT_NAME = "aqi_10p"

# Location Settings (Karachi)
LATITUDE = 24.86
LONGITUDE = 67.00
TIMEZONE = "Asia/Karachi"

# Feature Store Settings
FEATURE_GROUP_NAME = "karachi_weather_aqi_features"
FEATURE_GROUP_VERSION = 1

# Model Settings
MODEL_REGISTRY_NAME = "aqi_predictor"
TARGET_COLUMN = "pm25_next_hour"
PREDICTION_HORIZON = 24  # hours ahead to predict

# Feature columns
FEATURE_COLUMNS = [
    'temperature_2m', 
    'relative_humidity_2m', 
    'windspeed_10m',
    'pm2_5', 
    'pm10', 
    'pm25_change', 
    'temp_3h_avg',
    'pm25_lag1',
    'pm25_lag3',
    'pm25_lag6',
    'pm25_lag12',
    'hour', 
    'day', 
    'month', 
    'weekday'
]

# Data Collection Settings
HISTORICAL_MONTHS = 4  # Number of months of historical data to fetch

# Training Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42