import joblib
import shap
import matplotlib.pyplot as plt

from fetch_features import connect_to_hopsworks, load_data_from_feature_store, prepare_features_and_target

# Use trained model
model = joblib.load(r"C:\Users\Lenovo\Desktop\AQI_prediction\models\best_model_gradient_boosting.joblib")  # your Gradient Boosting model
# Load data
# Sample data (SHAP is expensive)
project = connect_to_hopsworks()
df = load_data_from_feature_store(project)
X, y, feature_names = prepare_features_and_target(df)
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

X_sample = X.sample(n=500, random_state=42)

# ----------------------------
# Create SHAP explainer
# ----------------------------
explainer = shap.Explainer(model, X_sample)

# Compute SHAP values
shap_values = explainer(X_sample)

# ----------------------------
# 1️⃣ SHAP Summary (Beeswarm)
# ----------------------------
shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)
plt.title("SHAP Summary Plot (Feature Impact)")
plt.savefig(f"training_results/shap_summary_plot.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# ----------------------------
# 2️⃣ SHAP Feature Importance (Bar)
# ----------------------------
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance")
plt.savefig(f"training_results/shap_feature_importance.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# ----------------------------
# 3️⃣ SHAP Scatter Plots (IMPORTANT FEATURES ONLY)
# ----------------------------
important_features = [
    "pm2_5",
    "pm25_change",
    "temp_3h_avg",
    "temperature_2m"
]

i=0
for feature in important_features:
    if feature in X_sample.columns:
        shap.plots.scatter(
            shap_values[:, feature],
            show=False
        )
        i+=1
        plt.title(f"SHAP Dependence Plot: {feature}")
        plt.tight_layout()
        plt.savefig(f"training_results/shap_{feature}.png", dpi=300, bbox_inches="tight")
        plt.show()
