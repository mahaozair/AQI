"""
AQI Prediction Dashboard
Real-time and forecasted Air Quality Index for Karachi
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import hopsworks
from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_COLUMNS
)

# Page configuration
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .hazard-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler from Hopsworks"""
    try:
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT_NAME,
            api_key_value=HOPSWORKS_API_KEY
        )
        
        mr = project.get_model_registry()
        model_obj = mr.get_model("aqi_predictor", version=1)
        model_dir = model_obj.download()
        
        model = joblib.load(f"{model_dir}/model.joblib")
        scaler = joblib.load(f"{model_dir}/scaler.joblib")
        
        return model, scaler, project
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


@st.cache_data(ttl=3600)
def load_latest_features(_project):
    """Load latest features from Feature Store"""
    try:
        fs = _project.get_feature_store()
        fg = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )
        
        df = fg.read()
        df = df.sort_values('event_time', ascending=False)
        
        return df
    except Exception as e:
        st.error(f"Error loading features: {e}")
        return None


def get_aqi_category(pm25):
    """Convert PM2.5 to AQI category"""
    if pm25 <= 12:
        return "Good", "#00e400"
    elif pm25 <= 35.4:
        return "Moderate", "#ffff00"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif pm25 <= 150.4:
        return "Unhealthy", "#ff0000"
    elif pm25 <= 250.4:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


def main():
    # Header
    st.markdown('<p class="main-header">🌫️ Karachi AQI Predictor</p>', unsafe_allow_html=True)
    st.markdown("### Real-time Air Quality Monitoring & 24-Hour Forecast")
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Flag_of_Pakistan.svg/320px-Flag_of_Pakistan.svg.png", width=100)
        st.title("⚙️ Settings")
        st.info("**Location:** Karachi, Pakistan")
        st.info(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        refresh = st.button("🔄 Refresh Data")
        
        st.markdown("---")
        st.markdown("### About AQI")
        st.markdown("""
        - **Good (0-12)**: Air quality is satisfactory
        - **Moderate (12-35)**: Acceptable quality
        - **Unhealthy for Sensitive (35-55)**: Sensitive groups may experience health effects
        - **Unhealthy (55-150)**: Everyone may begin to experience health effects
        - **Very Unhealthy (150-250)**: Health alert
        - **Hazardous (250+)**: Emergency conditions
        """)
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        model, scaler, project = load_model_and_scaler()
        
        if model is None or scaler is None or project is None:
            st.error("Failed to load model. Please check your configuration.")
            return
        
        df = load_latest_features(project)
        
        if df is None or len(df) == 0:
            st.error("No data available. Please run the feature pipeline first.")
            return
    
    # Get latest data
    latest = df.iloc[0]
    current_pm25 = latest.get('pm2_5', 0)
    current_category, current_color = get_aqi_category(current_pm25)
    
    # Current AQI Display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌡️ Current PM2.5")
        st.markdown(f'<div style="font-size: 3rem; font-weight: bold; color: {current_color};">{current_pm25:.1f}</div>', unsafe_allow_html=True)
        st.markdown(f"**{current_category}**")
    
    with col2:
        st.markdown("### 🌡️ Temperature")
        temp = latest.get('temperature_2m', 0)
        st.markdown(f'<div style="font-size: 3rem; font-weight: bold;">{temp:.1f}°C</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 💨 Wind Speed")
        wind = latest.get('windspeed_10m', 0)
        st.markdown(f'<div style="font-size: 3rem; font-weight: bold;">{wind:.1f} km/h</div>', unsafe_allow_html=True)
    
    # Hazard Alert
    if current_pm25 > 150:
        st.markdown(f"""
        <div class="hazard-alert">
        ⚠️ <strong>HEALTH ALERT:</strong> Air quality is {current_category}. 
        Avoid outdoor activities and keep windows closed.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Predictions
    st.markdown("### 🔮 24-Hour Forecast")
    
    # Prepare features for prediction
    features_available = [col for col in FEATURE_COLUMNS if col in df.columns]
    
    try:
        X = df[features_available].iloc[:24].values  # Last 24 hours
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Create forecast dataframe
        forecast_times = [latest['event_time'] + timedelta(hours=i) for i in range(24)]
        forecast_df = pd.DataFrame({
            'Time': forecast_times,
            'Predicted PM2.5': predictions
        })
        
        # Plot forecast
        fig = go.Figure()
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=forecast_df['Time'],
            y=forecast_df['Predicted PM2.5'],
            mode='lines+markers',
            name='Predicted PM2.5',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
        
        # Add AQI threshold lines
        fig.add_hline(y=12, line_dash="dash", line_color="green", annotation_text="Good")
        fig.add_hline(y=35.4, line_dash="dash", line_color="yellow", annotation_text="Moderate")
        fig.add_hline(y=55.4, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive")
        fig.add_hline(y=150.4, line_dash="dash", line_color="red", annotation_text="Unhealthy")
        
        fig.update_layout(
            title="24-Hour PM2.5 Forecast",
            xaxis_title="Time",
            yaxis_title="PM2.5 (μg/m³)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average (24h)", f"{predictions.mean():.1f} μg/m³")
        with col2:
            st.metric("Maximum (24h)", f"{predictions.max():.1f} μg/m³")
        with col3:
            st.metric("Minimum (24h)", f"{predictions.min():.1f} μg/m³")
        
    except Exception as e:
        st.error(f"Error generating predictions: {e}")
    
    st.markdown("---")
    
    # Historical Trends
    st.markdown("### 📊 Historical Trends (Last 7 Days)")
    
    # Get last 7 days of data
    recent_df = df.iloc[:168].copy()  # 7 days * 24 hours
    recent_df = recent_df.sort_values('event_time')
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["PM2.5 Trend", "Weather Correlation", "Hourly Patterns"])
    
    with tab1:
        fig = px.line(
            recent_df, 
            x='event_time', 
            y='pm2_5',
            title='PM2.5 Levels Over Time',
            labels={'pm2_5': 'PM2.5 (μg/m³)', 'event_time': 'Time'}
        )
        fig.add_hline(y=35.4, line_dash="dash", line_color="orange", annotation_text="Moderate Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.scatter(
            recent_df,
            x='temperature_2m',
            y='pm2_5',
            color='windspeed_10m',
            title='PM2.5 vs Temperature (colored by wind speed)',
            labels={
                'temperature_2m': 'Temperature (°C)',
                'pm2_5': 'PM2.5 (μg/m³)',
                'windspeed_10m': 'Wind Speed (km/h)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        hourly_avg = recent_df.groupby('hour')['pm2_5'].mean().reset_index()
        fig = px.bar(
            hourly_avg,
            x='hour',
            y='pm2_5',
            title='Average PM2.5 by Hour of Day',
            labels={'hour': 'Hour', 'pm2_5': 'Average PM2.5 (μg/m³)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    Built with ❤️ using Streamlit | Data from Open-Meteo API | Powered by Hopsworks
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()