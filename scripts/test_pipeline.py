"""
Unit tests for AQI Prediction pipelines
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_config_imports():
    """Test that config imports correctly"""
    from config import (
        HOPSWORKS_PROJECT_NAME,
        LATITUDE,
        LONGITUDE,
        FEATURE_COLUMNS,
        TARGET_COLUMN
    )
    
    assert HOPSWORKS_PROJECT_NAME is not None
    assert isinstance(LATITUDE, (int, float))
    assert isinstance(LONGITUDE, (int, float))
    assert len(FEATURE_COLUMNS) > 0
    assert TARGET_COLUMN is not None


def test_feature_columns_valid():
    """Test that feature columns are properly defined"""
    from config import FEATURE_COLUMNS
    
    # Check no duplicates
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))
    
    # Check all are strings
    assert all(isinstance(col, str) for col in FEATURE_COLUMNS)


def test_coordinates_valid():
    """Test that coordinates are within valid range"""
    from config import LATITUDE, LONGITUDE
    
    assert -90 <= LATITUDE <= 90
    assert -180 <= LONGITUDE <= 180


def test_sample_feature_computation():
    """Test feature computation logic"""
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'datetime': dates,
        'pm2_5': np.random.uniform(10, 100, 100),
        'temperature_2m': np.random.uniform(15, 35, 100)
    })
    
    # Test lag feature
    df['pm25_lag1'] = df['pm2_5'].shift(1)
    assert df['pm25_lag1'].isna().sum() == 1  # First value should be NaN
    
    # Test rolling average
    df['pm25_3h_avg'] = df['pm2_5'].rolling(window=3, min_periods=1).mean()
    assert len(df['pm25_3h_avg']) == len(df)
    
    # Test time features
    df['hour'] = df['datetime'].dt.hour
    assert df['hour'].min() >= 0 and df['hour'].max() <= 23


def test_target_variable_creation():
    """Test target variable (future PM2.5) creation"""
    df = pd.DataFrame({
        'pm2_5': [10, 20, 30, 40, 50]
    })
    
    df['pm25_next_hour'] = df['pm2_5'].shift(-1)
    
    # Check that future values are correct
    assert df.loc[0, 'pm25_next_hour'] == 20
    assert df.loc[1, 'pm25_next_hour'] == 30
    assert pd.isna(df.loc[4, 'pm25_next_hour'])  # Last value should be NaN


def test_model_metrics_range():
    """Test that model metrics are in expected ranges"""
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Simulate predictions
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 19, 31, 39, 51])
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    assert rmse >= 0  # RMSE must be non-negative
    assert -1 <= r2 <= 1  # R² typically in this range


def test_data_scaling():
    """Test that data scaling works correctly"""
    from sklearn.preprocessing import StandardScaler
    
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Check that scaled data has mean ≈ 0 and std ≈ 1
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)


def test_date_range_generation():
    """Test date range generation for backfill"""
    from config import HISTORICAL_MONTHS
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORICAL_MONTHS * 30)
    
    assert start_date < end_date
    diff_days = (end_date - start_date).days
    assert diff_days >= HISTORICAL_MONTHS * 30 - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])