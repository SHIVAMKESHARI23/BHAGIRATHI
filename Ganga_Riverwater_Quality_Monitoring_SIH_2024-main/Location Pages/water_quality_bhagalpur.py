import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import folium
from streamlit_folium import folium_static
from PIL import Image
import io
import base64

# Set page config (uncomment if needed)
# st.set_page_config(layout="wide")

# ==============================================
# MODIFICATIONS MADE:
# 1. Removed all API key dependencies
# 2. Added mock weather data generation
# 3. Added placeholder for Gemini reports
# 4. Improved visual design
# 5. Added local image fallback for maps
# ==============================================

# Locations dictionary with local data
LOCATIONS = {
    'GANGA AT U/S BHAGALPUR NEAR BARARIGHAT': {
        'file_path': "Bhagalpur.csv",  # Changed to local path
        'lat': 25.271603,
        'lon': 87.025665,
        'map_image': "map_placeholder.png"  # Add a local map image
    }  
}

# Generate mock weather data instead of API calls
def generate_mock_weather_forecast(start_date, days=5):
    """Generate realistic mock weather data"""
    forecasts = []
    for i in range(days):
        forecasts.append({
            'date': start_date + timedelta(days=i+1),
            'temperature': np.random.normal(25, 3),  # Around 25°C with variation
            'rainfall': max(0, np.random.normal(2, 1.5))  # Never negative
        })
    return forecasts

# Mock Gemini report generator
def generate_mock_ai_report(parameter, values, dates):
    """Generate a realistic water quality report without API"""
    report = f"""
    ## AI-Generated Water Quality Report for {parameter}
    
    **Forecast Summary:**
    - Average predicted value: {np.mean(values):.2f}
    - Trend: {'Increasing' if values[-1] > values[0] else 'Decreasing'}
    
    **Date-Specific Analysis:**
    """
    
    for date, value in zip(dates, values):
        report += f"\n- {date.strftime('%b %d')}: {value:.2f} ({'Normal' if value < np.mean(values) else 'Above normal'})"
    
    report += """
    
    **Recommendations:**
    - Monitor daily fluctuations
    - Compare with historical averages
    - Check for correlation with weather patterns
    
    *Note: This is a simulated report. For actual AI analysis, please configure the Gemini API.*
    """
    return report

# Modified greeting card with better visuals
def create_greeting_card():
    """Create an enhanced greeting card without external dependencies"""
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        theme = {
            "greeting": "Good Morning",
            "icon": "🌅",
            "color": "linear-gradient(135deg, #FF9A9E 0%, #FAD0C4 100%)"
        }
    elif 12 <= current_hour < 17:
        theme = {
            "greeting": "Good Afternoon", 
            "icon": "☀️",
            "color": "linear-gradient(135deg, #FFECD2 0%, #FCB69F 100%)"
        }
    elif 17 <= current_hour < 22:
        theme = {
            "greeting": "Good Evening",
            "icon": "🌇",
            "color": "linear-gradient(135deg, #A1C4FD 0%, #C2E9FB 100%)"
        }
    else:
        theme = {
            "greeting": "Good Night", 
            "icon": "🌙",
            "color": "linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%)"
        }
    
    st.markdown(f"""
    <div style="
        background: {theme['color']};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: {'white' if current_hour >= 22 or current_hour < 5 else '#333'};
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    ">
        <div style="font-size: 3rem; margin-bottom: 10px;">{theme['icon']}</div>
        <h2 style="margin: 0;">{theme['greeting']}</h2>
        <p>{datetime.now().strftime('%A, %B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

# Modified satellite map with fallback image
def create_location_map(latitude, longitude, location_name):
    """Create a map with fallback to local image"""
    try:
        m = folium.Map(
            location=[latitude, longitude],
            zoom_start=13,
            tiles="OpenStreetMap",  # Free tile layer
            control_scale=True
        )
        folium.Marker(
            [latitude, longitude],
            popup=location_name,
            icon=folium.Icon(color='blue')
        ).add_to(m)
        return m
    except:
        # Fallback to local image if map fails
        try:
            img_path = LOCATIONS[location_name]['map_image']
            st.image(img_path, caption=location_name, use_column_width=True)
            return None
        except:
            st.warning("Map display unavailable")
            return None

# Rest of your existing functions remain the same (prepare_input_data, make_donut, etc.)
# Just replace any API calls with mock data generators

def main():
    # Modern CSS styling
    st.markdown("""
    <style>
    /* Main page styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Tab styling */
    .stTabs [aria-selected="true"] {
        background-color: #3a7bd5 !important;
        color: white !important;
    }
    
    /* Donut container */
    .donut-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Page header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    ">
        <h1>Ganga River Water Quality Monitoring</h1>
        <p>Bhagalpur Monitoring Station</p>
    </div>
    """, unsafe_allow_html=True)

    # Hardcoded location since we're not using APIs
    selected_location = "GANGA AT U/S BHAGALPUR NEAR BARARIGHAT"
    
    if selected_location:
        # Load data - you'll need to have Bhagalpur.csv in your directory
        try:
            df = pd.read_csv(LOCATIONS[selected_location]['file_path'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Layout columns
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                create_greeting_card()
                
                # Weather widget with mock data
                st.markdown("""
                <div class="custom-card">
                    <h3>Weather Conditions</h3>
                    <p>🌡️ Temperature: 26.5°C</p>
                    <p>💧 Humidity: 65%</p>
                    <p>🌬️ Wind: 12 km/h NE</p>
                    <p>☔ Precipitation: 2mm</p>
                    <p><em>Simulated weather data</em></p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Map display
                st.markdown("### Monitoring Location")
                map_obj = create_location_map(
                    LOCATIONS[selected_location]['lat'],
                    LOCATIONS[selected_location]['lon'],
                    selected_location
                )
                if map_obj:
                    folium_static(map_obj, width=600)
            
            # Parameter tabs
            parameters = [col for col in df.columns if col not in ['Date', 'Temperature', 'Rainfall']]
            formatted_params = [p.replace('_', ' ').title() for p in parameters]
            
            tabs = st.tabs(formatted_params)
            
            for idx, param in enumerate(parameters):
                with tabs[idx]:
                    # Historical data
                    st.markdown(f"### Historical {param.replace('_', ' ').title()} Data")
                    st.line_chart(df.set_index('Date')[param])
                    
                    # Forecast section
                    st.markdown("### 5-Day Forecast")
                    
                    # Generate mock forecast
                    last_date = df['Date'].max()
                    forecast_dates = [last_date + timedelta(days=i+1) for i in range(5)]
                    forecast_values = df[param].iloc[-5:].values * np.random.normal(1, 0.1, 5)  # Random variation
                    
                    # Display forecast chart
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecast': forecast_values
                    })
                    st.line_chart(forecast_df.set_index('Date'))
                    
                    # Donut charts
                    st.markdown("### Daily Forecast")
                    st.markdown('<div class="donut-grid">', unsafe_allow_html=True)
                    
                    for date, value in zip(forecast_dates, forecast_values):
                        # You would call make_donut() here with your parameters
                        st.markdown(f"""
                        <div class="custom-card" style="text-align: center;">
                            <div style="font-size: 24px;">{value:.2f}</div>
                            <div>{date.strftime('%b %d')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # AI Report (mock)
                    st.markdown("### Water Quality Analysis")
                    st.markdown(generate_mock_ai_report(param, forecast_values, forecast_dates))
                    
        except FileNotFoundError:
            st.error("Data file not found. Please ensure Bhagalpur.csv exists in your directory.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()