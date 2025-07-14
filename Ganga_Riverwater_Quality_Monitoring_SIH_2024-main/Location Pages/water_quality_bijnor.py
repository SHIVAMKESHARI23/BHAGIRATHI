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
import base64
import io

# ==============================================
# KEY MODIFICATIONS:
# 1. Removed all API key dependencies
# 2. Added mock data generators for weather and AI reports
# 3. Improved visual design for Bijnor location
# 4. Added fallback mechanisms for all external services
# ==============================================

# Updated Locations dictionary for Bijnor
LOCATIONS = {
    'BIJNOR (U.P.)': {
        'file_path': "Bijnor.csv",  # Changed to local path
        'lat': 29.37,
        'lon': 78.14,
        'map_image': "bijnor_map.png"  # Add a local map image
    }  
}

# Mock weather data generator for Bijnor
def generate_bijnor_weather(start_date, days=5):
    """Generate realistic mock weather data for Bijnor"""
    base_temp = 25.0  # Average temperature for Bijnor
    forecasts = []
    for i in range(days):
        # Seasonal variation - hotter in summer (April-June)
        month = (start_date.month + i) % 12
        temp_variation = 0
        if 3 <= month <= 5:  # Summer
            temp_variation = 5
        elif 6 <= month <= 9:  # Monsoon
            temp_variation = -2
        
        forecasts.append({
            'date': start_date + timedelta(days=i+1),
            'temperature': base_temp + temp_variation + np.random.normal(0, 2),
            'rainfall': max(0, np.random.normal(2 if month >= 6 and month <= 9 else 0.5, 1))
        })
    return forecasts

# Mock AI report generator for water quality
def generate_mock_ai_report(parameter, values, dates, historical_data):
    """Generate a realistic water quality report without API"""
    # Calculate historical stats
    mean_val = historical_data[parameter].mean()
    std_val = historical_data[parameter].std()
    
    report = f"""
    ## Water Quality Report for {parameter.replace('_', ' ').title()}
    **Location:** Bijnor, Uttar Pradesh
    
    **Historical Baseline:**
    - Average: {mean_val:.2f}
    - Standard Deviation: {std_val:.2f}
    - Normal Range: {mean_val-std_val:.2f} to {mean_val+std_val:.2f}
    
    **5-Day Forecast Analysis:**
    """
    
    for date, value in zip(dates, values):
        deviation = (value - mean_val) / std_val
        if deviation > 1.5:
            status = "⚠️ Significantly High"
        elif deviation > 0.5:
            status = "↑ Moderately High"
        elif deviation < -1.5:
            status = "⚠️ Significantly Low" 
        elif deviation < -0.5:
            status = "↓ Moderately Low"
        else:
            status = "✅ Normal Range"
        
        report += f"\n- {date.strftime('%b %d')}: {value:.2f} ({status})"
    
    report += """
    
    **Recommendations:**
    - Monitor daily fluctuations
    - Compare with historical averages
    - Check nearby industrial activity for potential pollution sources
    - Increase sampling frequency if values approach warning thresholds
    
    *Note: This is a simulated report. For actual AI analysis, please configure the Gemini API.*
    """
    return report

# Enhanced greeting card for Bijnor
def create_greeting_card():
    """Create location-specific greeting card"""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        theme = {
            "greeting": "Good Morning from Bijnor",
            "icon": "🌅",
            "color": "linear-gradient(135deg, #FF9A9E 0%, #FAD0C4 100%)"
        }
    elif 12 <= current_hour < 17:
        theme = {
            "greeting": "Good Afternoon in Bijnor", 
            "icon": "🌞",
            "color": "linear-gradient(135deg, #FFB347 0%, #FFCC33 100%)"
        }
    elif 17 <= current_hour < 22:
        theme = {
            "greeting": "Good Evening in Bijnor",
            "icon": "🌇",
            "color": "linear-gradient(135deg, #4776E6 0%, #8E54E9 100%)"
        }
    else:
        theme = {
            "greeting": "Good Night from Bijnor", 
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
        <p style="font-size: 0.8rem; opacity: 0.8;">Ganga River Monitoring Station</p>
    </div>
    """, unsafe_allow_html=True)

# Modified map display for Bijnor
def create_bijnor_map():
    """Create a map with fallback to local image"""
    try:
        m = folium.Map(
            location=[29.37, 78.14],
            zoom_start=12,
            tiles="OpenStreetMap",  # Free tile layer
            control_scale=True
        )
        folium.Marker(
            [29.37, 78.14],
            popup="Bijnor Monitoring Station",
            icon=folium.Icon(color='blue', icon='tint', prefix='fa')
        ).add_to(m)
        
        # Add Ganga river overlay
        folium.PolyLine(
            [[29.37, 78.1], [29.37, 78.18]],  # Simplified river path
            color="blue",
            weight=5,
            opacity=0.7
        ).add_to(m)
        
        return m
    except:
        # Fallback to local image if map fails
        try:
            st.image(LOCATIONS['BIJNOR (U.P.)']['map_image'], 
                   caption="Bijnor Monitoring Location",
                   use_column_width=True)
            return None
        except:
            st.warning("Map display unavailable. Please check your internet connection.")
            return None

# Modified weather display for Bijnor
def create_bijnor_weather_kpi():
    """Create weather display with mock data"""
    # Generate mock weather data
    current_weather = {
        'temperature': 26.5 + np.random.normal(0, 2),
        'humidity': 65 + np.random.normal(0, 5),
        'wind_speed': 12 + np.random.normal(0, 3),
        'precipitation': max(0, np.random.normal(1, 0.5))
    }
    
    # Create Altair chart
    weather_data = pd.DataFrame([
        {'Metric': 'Temperature', 'Value': current_weather['temperature'], 'Unit': '°C'},
        {'Metric': 'Humidity', 'Value': current_weather['humidity'], 'Unit': '%'},
        {'Metric': 'Wind Speed', 'Value': current_weather['wind_speed'], 'Unit': 'km/h'},
        {'Metric': 'Precipitation', 'Value': current_weather['precipitation'], 'Unit': 'mm/h'}
    ])
    
    chart = alt.Chart(weather_data).mark_bar().encode(
        x='Metric:N',
        y='Value:Q',
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=['Temperature', 'Humidity', 'Wind Speed', 'Precipitation'],
            range=['#FF6B6B', '#4ECDC4', '#2575FC', '#8A4FFF']
        )),
        tooltip=['Metric', 'Value', 'Unit']
    ).properties(
        title='Current Weather Conditions in Bijnor',
        width=400,
        height=250
    )
    
    return chart

# Rest of your existing functions (prepare_input_data, make_donut, etc.) remain the same
# Just replace any API calls with the mock data generators

def main():
    # Modern CSS styling for Bijnor theme
    st.markdown("""
    <style>
    /* Main page styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .bijnor-header {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        padding: 25px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Card styling */
    .bijnor-card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #3a7bd5;
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
    
    /* Parameter value styling */
    .param-value {
        font-size: 1.2rem;
        font-weight: bold;
        color: #3a7bd5;
    }
    </style>
    """, unsafe_allow_html=True)

    # Page header with Bijnor focus
    st.markdown("""
    <div class="bijnor-header">
        <h1>Ganga River Water Quality Monitoring</h1>
        <h3>Bijnor, Uttar Pradesh</h3>
        <p>Real-time monitoring and forecasting system</p>
    </div>
    """, unsafe_allow_html=True)

    # Hardcoded location for Bijnor
    selected_location = "BIJNOR (U.P.)"
    
    if selected_location:
        # Load data - you'll need to have Bijnor.csv in your directory
        try:
            df = pd.read_csv(LOCATIONS[selected_location]['file_path'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Layout columns
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                create_greeting_card()
                
                # Weather widget with mock data
                st.altair_chart(create_bijnor_weather_kpi(), use_container_width=True)
            
            with col2:
                # Map display
                st.markdown("### Monitoring Location")
                map_obj = create_bijnor_map()
                if map_obj:
                    folium_static(map_obj, width=600)
            
            # Parameter tabs
            parameters = [col for col in df.columns if col not in ['Date', 'Temperature', 'Rainfall']]
            formatted_params = [p.replace('_', ' ').title() for p in parameters]
            
            tabs = st.tabs(formatted_params)
            
            for idx, param in enumerate(parameters):
                with tabs[idx]:
                    try:
                        # Historical data
                        st.markdown(f"### Historical {param.replace('_', ' ').title()} Data")
                        st.line_chart(df.set_index('Date')[param])
                        
                        # Forecast section
                        st.markdown("### 5-Day Forecast")
                        
                        # Generate mock forecast
                        last_date = df['Date'].max()
                        weather_forecast = generate_bijnor_weather(last_date)
                        
                        # Load model and predict (using your existing functions)
                        model = load_model_for_parameter(param)
                        if model:
                            X, X_exo, scaler, last_10_days, temp_scaler, rainfall_scaler = prepare_input_data(
                                df, weather_forecast, param
                            )
                            prediction = model.predict([X, X_exo])
                            forecast_values = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
                            forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast_values))]
                            
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
                                donut = make_donut(
                                    input_response=value,
                                    input_text=date.strftime('%b %d'),
                                    parameter=param,
                                    parameter_data=df,
                                    selected_parameter=param
                                )
                                st.altair_chart(donut, use_container_width=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # AI Report (mock)
                            st.markdown("### Water Quality Analysis")
                            st.markdown(generate_mock_ai_report(param, forecast_values, forecast_dates, df))
                            
                            # Water quality alerts
                            display_water_quality_alerts({param: forecast_values[-1]}, df)
                            
                    except Exception as e:
                        st.error(f"Error processing {param}: {str(e)}")
                        
        except FileNotFoundError:
            st.error("Data file not found. Please ensure Bijnor.csv exists in your directory.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()