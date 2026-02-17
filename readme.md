Project Objective:

The goal of this project is to build a complete AQI forecasting system that:

Fetches raw weather and pollutant data

Performs feature engineering

Trains multiple machine learning models

Compares performance across models

Selects the best-performing model

Automates feature and training pipelines

Serves predictions through an interactive dashboard

Data Source

Data was collected using the Open-Meteo API, which provides:

Weather data (temperature, humidity, wind speed, pressure)

Air quality data (PM2.5 and related pollutants)

The system fetches both historical and real-time data to support training and forecasting

PM2.5 was selected as the main target variable because:

In dense urban cities like Karachi, PM2.5 strongly influences AQI.

It is the most health-relevant pollutant.

It frequently dominates other pollutant indicators in AQI calculation.

It has strong predictive power for overall air quality.

By accurately predicting PM2.5 levels, we effectively predict AQI trends.

Features:
Engineered Features:

The following features were created:

Hour

Day

Month

Rolling averages

AQI change rate

Derived meteorological interactions

These features allow the model to capture:

Seasonal trends

Daily pollution cycles

Sudden spikes

Long-term patterns

Model Training & Evaluation

We experimented with 7 different regression models:

Gradient Boosting

LightGBM

XGBoost

Random Forest

Lasso

Ridge

ElasticNet

The following metrics were used to compare models:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

MAPE (Mean Absolute Percentage Error)

Final Model Selection

Gradient Boosting Regressor was selected as the primary production model because:

It achieved the highest R² score (0.9526).

It had the lowest RMSE.

It provided strong generalization across validation splits.

It showed stable error distribution compared to other models.

Although LightGBM performed similarly, Gradient Boosting slightly outperformed it in stability and error metrics.

Model Explainability

To improve transparency:

SHAP was used for feature importance analysis.

The system identifies dominant factors influencing PM2.5 levels.

Helps interpret pollution spikes and seasonal changes.

The system flags AQI categories such as:

Moderate

Unhealthy

Very Unhealthy

Hazardous