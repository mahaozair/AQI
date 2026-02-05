"""
Feature Pipeline for AQI Prediction System
Fetches weather and air quality data, computes features, and stores in Hopsworks Feature Store
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import hopsworks
from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    HISTORICAL_MONTHS
)


def fetch_open_meteo_data(start_date, end_date):
    """
    Fetch weather and air quality data from Open-Meteo API (Free)
    Only fetches fields that match your existing Feature Group schema
    """
    print("\n" + "="*70)
    print("FETCHING DATA FROM OPEN-METEO API")
    print("="*70)
    print(f"Location: Karachi ({LATITUDE}, {LONGITUDE})")
    print(f"Date range: {start_date} to {end_date}")
    
    # Format dates for API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Open-Meteo API endpoints
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    air_quality_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    # Weather parameters - ONLY what you need
    weather_params = {
        'latitude': LATITUDE,
        'longitude': LONGITUDE,
        'start_date': start_str,
        'end_date': end_str,
        'hourly': [
            'temperature_2m',
            'relative_humidity_2m',
            'windspeed_10m'
        ],
        'timezone': TIMEZONE
    }
    
    # Air quality parameters - ONLY what you need
    air_params = {
        'latitude': LATITUDE,
        'longitude': LONGITUDE,
        'start_date': start_str,
        'end_date': end_str,
        'hourly': [
            'pm10',
            'pm2_5'
        ],
        'timezone': TIMEZONE
    }
    
    try:
        # Fetch weather data
        print("\nFetching weather data...")
        weather_response = requests.get(weather_url, params=weather_params, timeout=30)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        # Fetch air quality data
        print("Fetching air quality data...")
        air_response = requests.get(air_quality_url, params=air_params, timeout=30)
        air_response.raise_for_status()
        air_data = air_response.json()
        
        # Create DataFrames - ONLY fields that exist in your schema
        weather_df = pd.DataFrame({
            'datetime': pd.to_datetime(weather_data['hourly']['time']),
            'temperature_2m': weather_data['hourly']['temperature_2m'],
            'relative_humidity_2m': weather_data['hourly']['relative_humidity_2m'],
            'windspeed_10m': weather_data['hourly']['windspeed_10m']
        })
        
        air_df = pd.DataFrame({
            'datetime': pd.to_datetime(air_data['hourly']['time']),
            'pm10': air_data['hourly']['pm10'],
            'pm2_5': air_data['hourly']['pm2_5']
        })
        
        # Merge datasets
        df = pd.merge(weather_df, air_df, on='datetime', how='inner')
        
        print(f"✅ Successfully fetched {len(df)} hourly records")
        print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"   Columns: {list(df.columns)}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        raise
    except KeyError as e:
        print(f"❌ Error parsing API response: {e}")
        raise


def compute_features(df):
    """
    Compute engineered features from raw data
    ONLY generates features that exist in your Feature Group
    """
    print("\n" + "="*70)
    print("COMPUTING FEATURES")
    print("="*70)
    
    df = df.copy()
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 1. Time-based features (only the ones you have)
    print("Computing time-based features...")
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.dayofweek
    
    # 2. Rolling statistics (only the ones you have)
    print("Computing rolling statistics...")
    df['temp_3h_avg'] = df['temperature_2m'].rolling(window=3, min_periods=1).mean()
    
    # 3. Change rates (only the ones you have)
    print("Computing change rates...")
    df['pm25_change'] = df['pm2_5'].diff()
    
    # 4. Target variable (PM2.5 next hour - what we want to predict)
    print("Computing target variable...")
    df['pm25_next_hour'] = df['pm2_5'].shift(-1)
    
    print(f"✅ Computed features matching your existing Feature Group schema")
    print(f"   Features generated: hour, day, month, weekday, temp_3h_avg, pm25_change, pm25_next_hour")
    
    return df


def connect_to_hopsworks():
    """Connect to Hopsworks Feature Store"""
    print("\n" + "="*70)
    print("CONNECTING TO HOPSWORKS")
    print("="*70)
    
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )
    
    print(f"✅ Connected to project: {HOPSWORKS_PROJECT_NAME}")
    return project


def save_to_feature_store(df, project, mode='append'):
    """
    Save features to Hopsworks Feature Store
    mode: 'append' or 'overwrite'
    """
    print("\n" + "="*70)
    print("SAVING TO FEATURE STORE")
    print("="*70)
    
    fs = project.get_feature_store()
    
    # Prepare dataframe
    df_to_save = df.copy()
    
    # Remove rows with NaN in target variable
    initial_rows = len(df_to_save)
    df_to_save = df_to_save.dropna(subset=['pm25_next_hour'])
    print(f"Removed {initial_rows - len(df_to_save)} rows with NaN target")
    
    # Convert datetime to timestamp for Hopsworks
    df_to_save['event_time'] = pd.to_datetime(df_to_save['datetime'])
    
    # CRITICAL: Only keep features that exist in your existing Feature Group
    # Based on your feature_names.json
    EXISTING_FEATURES = [
        'temperature_2m',
        'relative_humidity_2m',
        'windspeed_10m',
        'pm2_5',
        'pm10',
        'pm25_change',
        'temp_3h_avg',
        'hour',
        'day',
        'month',
        'weekday',
        'pm25_next_hour',  # target
        'event_time'  # timestamp
    ]
    
    # Filter to only existing features
    available_cols = [col for col in EXISTING_FEATURES if col in df_to_save.columns]
    df_to_save = df_to_save[available_cols]
    
    print(f"Using {len(available_cols)} features that match existing Feature Group schema")
    print(f"Features: {available_cols}")
    
    print(f"\nSaving {len(df_to_save)} rows to Feature Store...")
    print(f"Feature Group: {FEATURE_GROUP_NAME}")
    
    try:
        # Get existing feature group (don't create new one - use yours)
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        
        print(f"✅ Found existing feature group: {fg.name} (version {fg.version})")
        
        # Insert data
        if mode == 'overwrite':
            print("⚠️  WARNING: Overwrite mode will replace ALL your existing data!")
            fg.insert(df_to_save, overwrite=True)
            print(f"✅ Overwrote feature group with {len(df_to_save)} rows")
        else:
            fg.insert(df_to_save, overwrite=False)
            print(f"✅ Appended {len(df_to_save)} rows to feature group")
        
        print(f"\n📊 Feature Store Statistics:")
        print(f"   Total features: {len(df_to_save.columns)}")
        print(f"   Date range: {df_to_save['event_time'].min()} to {df_to_save['event_time'].max()}")
        
    except Exception as e:
        print(f"❌ Error saving to Feature Store: {e}")
        raise


def backfill_historical_data():
    """
    Backfill historical data for initial model training
    """
    print("\n" + "="*70)
    print("BACKFILLING HISTORICAL DATA")
    print("="*70)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORICAL_MONTHS * 30)
    
    print(f"Backfilling {HISTORICAL_MONTHS} months of data")
    print(f"From: {start_date.date()}")
    print(f"To: {end_date.date()}")
    
    # Connect to Hopsworks
    project = connect_to_hopsworks()
    
    # Fetch data
    df = fetch_open_meteo_data(start_date, end_date)
    
    # Compute features
    df = compute_features(df)
    
    # Save to Feature Store (overwrite mode for backfill)
    save_to_feature_store(df, project, mode='overwrite')
    
    print("\n" + "="*70)
    print("✅ BACKFILL COMPLETED")
    print("="*70)


def update_latest_data():
    """
    Fetch and update latest data (for hourly CI/CD runs)
    """
    print("\n" + "="*70)
    print("UPDATING LATEST DATA")
    print("="*70)
    
    # Fetch last 7 days to ensure we have overlap and can compute features
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Fetching latest data from last 7 days")
    print(f"From: {start_date.date()}")
    print(f"To: {end_date.date()}")
    
    # Connect to Hopsworks
    project = connect_to_hopsworks()
    
    # Fetch data
    df = fetch_open_meteo_data(start_date, end_date)
    
    # Compute features
    df = compute_features(df)
    
    # Save to Feature Store (append mode for updates)
    save_to_feature_store(df, project, mode='append')
    
    print("\n" + "="*70)
    print("✅ UPDATE COMPLETED")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--backfill':
        # Run backfill for historical data
        backfill_historical_data()
    else:
        # Run regular update for latest data
        update_latest_data()