
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from config import TEST_SIZE
from fetch_features import RANDOM_STATE, TARGET_COLUMN, connect_to_hopsworks, load_data_from_feature_store, prepare_features_and_target
import xgboost as xgb
import lightgbm as lgb
import joblib
import hopsworks
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def split_and_scale_data(X, y):
    """Split into train/test and scale features"""
    print("\n" + "="*70)
    print("SPLITTING AND SCALING DATA")
    print("="*70)
    
    # Split data (time series, so no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=False  # Keep time series order
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Data scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare"""
    print("\n" + "="*70)
    print("TRAINING MULTIPLE MODELS")
    print("="*70)
    
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Lasso': Lasso(alpha=0.1, random_state=RANDOM_STATE),
        'ElasticNet': ElasticNet(alpha=0.1, random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[name] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        trained_models[name] = model
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
    
    return results, trained_models


def save_results(results, trained_models, scaler, feature_names, project):
    """Save models and results locally AND to Hopsworks Model Registry"""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('training_results', exist_ok=True)
    
    # Find best model
    best_name = min(results.keys(), key=lambda x: results[x]['rmse'])
    best_model = trained_models[best_name]
    best_metrics = results[best_name]
    
    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   RMSE: {best_metrics['rmse']:.4f}")
    print(f"   MAE:  {best_metrics['mae']:.4f}")
    print(f"   R²:   {best_metrics['r2']:.4f}")
    print(f"   MAPE: {best_metrics['mape']:.2f}%")
    
    # Save locally first
    model_file = f"models/best_model_{best_name.lower().replace(' ', '_')}.joblib"
    joblib.dump(best_model, model_file)
    print(f"\n✅ Best model saved locally: {model_file}")
    
    scaler_file = "models/scaler.joblib"
    joblib.dump(scaler, scaler_file)
    print(f"✅ Scaler saved locally: {scaler_file}")
    
    feature_file = "models/feature_names.json"
    with open(feature_file, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✅ Feature names saved: {feature_file}")
    
    # Save results
    results_file = f"training_results/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved: {results_file}")
    
    # Create comparison plot
    results_df = pd.DataFrame(results).T.sort_values('rmse')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # RMSE
    axes[0, 0].barh(results_df.index, results_df['rmse'], color='steelblue')
    axes[0, 0].set_xlabel('RMSE (lower is better)')
    axes[0, 0].set_title('Root Mean Squared Error')
    axes[0, 0].invert_yaxis()
    
    # MAE
    axes[0, 1].barh(results_df.index, results_df['mae'], color='coral')
    axes[0, 1].set_xlabel('MAE (lower is better)')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].invert_yaxis()
    
    # R²
    axes[1, 0].barh(results_df.index, results_df['r2'], color='mediumseagreen')
    axes[1, 0].set_xlabel('R² Score (higher is better)')
    axes[1, 0].set_title('R² Score')
    axes[1, 0].invert_yaxis()
    
    # MAPE
    axes[1, 1].barh(results_df.index, results_df['mape'], color='orchid')
    axes[1, 1].set_xlabel('MAPE % (lower is better)')
    axes[1, 1].set_title('Mean Absolute Percentage Error')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    plot_file = "training_results/model_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved: {plot_file}")
    plt.close()
    
    # Save comparison CSV
    csv_file = "training_results/model_comparison.csv"
    results_df.to_csv(csv_file)
    print(f"✅ Comparison CSV saved: {csv_file}")
    
    # ========================================================================
    # SAVE TO HOPSWORKS MODEL REGISTRY (for CI/CD deployment)
    # ========================================================================
    print("\n" + "="*70)
    print("SAVING TO HOPSWORKS MODEL REGISTRY")
    print("="*70)
    
    try:
        # Get model registry
        mr = project.get_model_registry()
        
        # Create model directory with all artifacts
        model_dir = "aqi_model_artifacts"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save all artifacts to model directory
        joblib.dump(best_model, f"{model_dir}/model.joblib")
        joblib.dump(scaler, f"{model_dir}/scaler.joblib")
        with open(f"{model_dir}/feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        with open(f"{model_dir}/metrics.json", 'w') as f:
            json.dump(best_metrics, f, indent=2)
        
        # Create input/output schema
        input_schema = {
            "features": feature_names,
            "type": "float64"
        }
        output_schema = {
            "prediction": TARGET_COLUMN,
            "type": "float64"
        }
        
        # Create model in registry
        aqi_model = mr.python.create_model(
            name="aqi_predictor",
            description=f"AQI Prediction model using {best_name} - Predicts PM2.5 24h ahead",
            metrics={
                "rmse": float(best_metrics['rmse']),
                "mae": float(best_metrics['mae']),
                "r2": float(best_metrics['r2']),
                "mape": float(best_metrics['mape'])
            },
            model_schema={
                "input_schema": input_schema,
                "output_schema": output_schema
            }
        )
        
        # Save the model with all artifacts
        aqi_model.save(model_dir)
        
        print(f"\n🎉 MODEL REGISTERED TO HOPSWORKS!")
        print(f"   Name: aqi_predictor")
        print(f"   Algorithm: {best_name}")
        print(f"   Version: {aqi_model.version}")
        print(f"   RMSE: {best_metrics['rmse']:.4f}")
        print(f"   R²: {best_metrics['r2']:.4f}")
        print(f"\n✅ Model is now ready for CI/CD deployment!")
        
    except Exception as e:
        print(f"\n⚠️  Warning: Could not save to Model Registry: {e}")
        print("Model saved locally but not in Hopsworks Model Registry.")
        print("You can manually register it later or check your permissions.")
    
    return best_name, best_model, best_metrics


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("AQI PREDICTION - TRAINING PIPELINE")
    print("Fetching data from Hopsworks Feature Store")
    print("="*70)
    
    start_time = datetime.now()
    
    try:
        # 1. Connect to Hopsworks
        project = connect_to_hopsworks()
        
        # 2. Load data from Feature Store
        df = load_data_from_feature_store(project)
        
        # 3. Prepare features and target
        X, y, feature_names = prepare_features_and_target(df)
        
        # 4. Split and scale data
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
        
        # 5. Train models
        results, trained_models = train_models(X_train, y_train, X_test, y_test)
        
        # 6. Save everything (locally + Hopsworks Model Registry)
        best_name, best_model, best_metrics = save_results(results, trained_models, scaler, feature_names, project)
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print("\n" + "="*70)
        print("TRAINING PIPELINE COMPLETED")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TRAINING FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. Your Hopsworks API key and project name are correct")
        print("2. The feature group exists in Hopsworks")
        print("3. The target column name matches what you stored")


if __name__ == "__main__":
    main()