import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgb
import joblib
import hopsworks
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Import configurations
from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION ,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)

def connect_to_hopsworks():
    """Connect to Hopsworks"""
    print("\n" + "="*70)
    print("CONNECTING TO HOPSWORKS")
    print("="*70)
    
    project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,  # Replace with your project name
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value=HOPSWORKS_API_KEY,
    engine="python"# Get from Hopsworks UI > Account Settings > API Keys
    )
    
    print(f"✅ Connected to project: {HOPSWORKS_PROJECT_NAME}\n")
    return project


def load_data_from_feature_store(project):
    """Load data from Hopsworks Feature Store"""
    print("\n" + "="*70)
    print("LOADING DATA FROM FEATURE STORE")
    print("="*70)
    
    fs = project.get_feature_store()
    
    # Get the feature group
    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )
    
    print(f"Feature Group: {fg.name} (version {fg.version})")
    
    # Read all data
    df = fg.read()
    
    print(f"✅ Loaded {len(df)} rows from Feature Store")
    print(f"   Columns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    
    return df


def prepare_features_and_target(df):
    """Prepare X and y for training"""
    print("\n" + "="*70)
    print("PREPARING FEATURES AND TARGET")
    print("="*70)
    
    # Check which features are available
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Warning: Missing features: {missing_features}")
        print(f"Using available features only: {available_features}")
    
    # Check target column
    if TARGET_COLUMN not in df.columns:
        print(f"\n❌ Error: Target column '{TARGET_COLUMN}' not found!")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")
    
    # Prepare X and y
    X = df[available_features].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Remove rows with NaN
    print(f"\nBefore removing NaN:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"\nAfter removing NaN:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: {list(X.columns)}")
    
    return X, y, available_features
