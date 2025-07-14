import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import MinMaxScaler
import altair as alt
import matplotlib.dates as mdates
import folium
from streamlit_folium import folium_static
from matplotlib import gridspec
from matplotlib.patches import Arc

# Remove API key dependencies and add dummy data functions
LOCATIONS = {
    'NAINI': {
        'file_path': r"Datasets\Naini.csv",
        'lat': 25.3871,
        'lon': 81.8597
    }  
}

def get_time_based_greeting():
    """Determine greeting based on current time with enhanced details."""
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        return {
            "greeting": "Good Morning",
            "bg_color": "#FFF4E0",
            "accent_color": "#FFD700", 
            "text_color": "#333333",
            "icon": "☀️",
            "message": "Rise and shine! A brand new day awaits your brilliance.",
            "gradient_start": "#FFFFFF",
            "gradient_end": "#FFE5B4"
        }
    elif 12 <= current_hour < 17:
        return {
            "greeting": "Good Afternoon", 
            "bg_color": "#E6F3FF",
            "accent_color": "#87CEEB",
            "text_color": "#2C3E50", 
            "icon": "☕",
            "message": "Halfway through the day. Keep pushing forward!",
            "gradient_start": "#F0F8FF",
            "gradient_end": "#87CEEB"
        }
    elif 17 <= current_hour < 22:
        return {
            "greeting": "Good Evening",
            "bg_color": "#E6E6FA", 
            "accent_color": "#DDA0DD",
            "text_color": "#4A4A4A",
            "icon": "🌆",
            "message": "Time to unwind and reflect on your day's achievements.",
            "gradient_start": "#F0E6FF", 
            "gradient_end": "#DDA0DD"
        }
    else:
        return {
            "greeting": "Good Night", 
            "bg_color": "#191970", 
            "accent_color": "#4169E1",
            "text_color": "#FFFFFF",
            "icon": "🌙",
            "message": "Rest well. Tomorrow brings new opportunities.",
            "gradient_start": "#000080", 
            "gradient_end": "#191970"
        }

def create_custom_css():
    """Create custom CSS for improved styling and hover effects."""
    return f"""
    <style>
    .greeting-card {{
        background: linear-gradient(135deg, 
            {st.session_state.time_details['gradient_start']} 0%, 
            {st.session_state.time_details['gradient_end']} 100%);
        border-radius: 10px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
        color: {st.session_state.time_details['text_color']};
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
    }}
    
    .greeting-card:hover {{
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }}
    
    .greeting-icon {{
        font-size: 3rem;
        margin-bottom: 10px;
        animation: float 2s ease-in-out infinite;
    }}
    
    .greeting-title {{
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: {st.session_state.time_details['accent_color']};
    }}
    
    .greeting-date {{
        font-size: 1rem;
        color: {st.session_state.time_details['text_color']};
        margin-bottom: 10px;
    }}
    
    .greeting-message {{
        font-size: 0.9rem;
        font-style: italic;
        opacity: 0.8;
        margin-top: 10px;
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
        100% {{ transform: translateY(0px); }}
    }}
    </style>
    """

def create_greeting_card():
    """Create an enhanced, interactive greeting card using Streamlit."""
    st.session_state.time_details = get_time_based_greeting()
    st.markdown(create_custom_css(), unsafe_allow_html=True)
    st.markdown(f'''
    <div class="greeting-card">
        <div class="greeting-icon">{st.session_state.time_details['icon']}</div>
        <div class="greeting-title">{st.session_state.time_details['greeting']}</div>
        <div class="greeting-date">{datetime.now().strftime('%A, %B %d, %Y')}</div>
    ''', unsafe_allow_html=True)

def create_satellite_map(latitude, longitude):
    """Create a Folium map with OpenStreetMap as fallback"""
    m = folium.Map(
        location=[latitude, longitude], 
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    folium.Marker(
        [latitude, longitude],
        popup='Monitoring Location',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    return m

def parse_date(date_str):
    """Flexible date parsing function"""
    date_formats = [
        '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d', 
        '%d/%m/%Y', '%m/%d/%Y'
    ]
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    raise ValueError(f"Unable to parse date: {date_str}")

def prepare_input_data(historical_data, weather_forecast, parameter):
    """Prepare input data for model prediction with fixed sequence length"""
    historical_data['Date'] = historical_data['Date'].apply(parse_date)
    historical_data = historical_data.dropna(subset=['Date'])
    historical_data.set_index('Date', inplace=True)
    
    last_10_days = historical_data[parameter].tail(10)
    
    if len(weather_forecast) < 5:
        while len(weather_forecast) < 5:
            last_forecast = weather_forecast[-1]
            weather_forecast.append({
                'date': last_forecast['date'] + timedelta(days=1),
                'temperature': last_forecast['temperature'],
                'rainfall': last_forecast['rainfall']
            })
    elif len(weather_forecast) > 5:
        weather_forecast = weather_forecast[:5]
    
    temps = [w['temperature'] for w in weather_forecast]
    rainfalls = [w['rainfall'] for w in weather_forecast]
    
    param_scaler = MinMaxScaler()
    scaled_data = param_scaler.fit_transform(last_10_days.values.reshape(-1, 1))
    
    temp_scaler = MinMaxScaler()
    rainfall_scaler = MinMaxScaler()
    
    scaled_temps = temp_scaler.fit_transform(np.array(temps).reshape(-1, 1))
    scaled_rainfalls = rainfall_scaler.fit_transform(np.array(rainfalls).reshape(-1, 1))
    
    scaled_exogenous = np.column_stack([scaled_temps.flatten(), scaled_rainfalls.flatten()])
    
    X = scaled_data.reshape(1, 10, 1)
    X_exo = scaled_exogenous.reshape(1, 5, 2)
    
    return X, X_exo, param_scaler, last_10_days, temp_scaler, rainfall_scaler

def generate_dummy_weather_forecast(start_date):
    """Generate dummy weather forecast data for Naini"""
    return [
        {
            'date': start_date + timedelta(days=i+1),
            'temperature': np.random.uniform(22, 35),  # Typical Naini temperatures
            'rainfall': np.random.uniform(0, 10) if i < 2 else 0  # More likely rain in first 2 days
        } for i in range(5)
    ]

def create_altair_forecast_plot(historical_data, forecast_data, parameter):
    """Create an Altair plot showing historical and forecasted data"""
    if not isinstance(historical_data.index, pd.DatetimeIndex):
        historical_data.index = pd.to_datetime(historical_data.index)
    
    historical_df = pd.DataFrame({
        'Date': historical_data.index,
        'Value': historical_data.values,
        'Type': 'Historical'
    })
    
    forecast_dates = [historical_data.index[-1] + timedelta(days=i+1) for i in range(len(forecast_data))]
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Value': forecast_data,
        'Type': 'Forecast'
    })
    
    combined_df = pd.concat([historical_df, forecast_df])
    
    chart = alt.Chart(combined_df).mark_line(point=True).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Value:Q', title=parameter),
        color=alt.Color('Type:N', 
            scale=alt.Scale(domain=['Historical', 'Forecast'], 
                            range=['steelblue', 'red']),
            legend=alt.Legend(title='Data Type')
        ),
        tooltip=['Date:T', 'Value:Q', 'Type:N']
    ).properties(
        title=f'{parameter} - Historical and Forecast',
        width=700,
        height=400
    ).interactive()
    
    return chart

def load_model_for_parameter(parameter):
    """Load the pre-trained model for a specific water quality parameter."""
    parameter_model_paths = {
        "Biochemical Oxygen Demand": r"models\Biochemical_Oxygen_Demand_water_quality_lstm_model.keras",
        "Dissolved Oxygen": r"models\Dissolved_Oxygen_water_quality_lstm_model.keras",
        "pH": r"models\pH_water_quality_lstm_model.keras",
        "Turbidity": r"models\Turbidity_water_quality_lstm_model.keras",
        "Nitrate": r"models\Nitrate_water_quality_lstm_model.keras",
        "Fecal Coliform": r"models\Fecal_Coliform_water_quality_lstm_model.keras",
        "Fecal Streptococci": r"models\Fecal_Streptococci_water_quality_lstm_model.keras",
        "Total Coliform": r"models\Total_Coliform_water_quality_lstm_model.keras",
        "Conductivity": r"models\Conductivity_water_quality_lstm_model.keras"
    }
    model_path = parameter_model_paths.get(parameter)
    
    if model_path and os.path.exists(model_path):
        return load_model(model_path)
    else:
        st.error(f"Model for {parameter} not found.")
        return None

def create_altair_historical_plot(df, parameter):
    """Create an Altair plot of historical data"""
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y(f'{parameter}:Q', title=parameter),
        tooltip=['Date:T', f'{parameter}:Q']
    ).properties(
        title=f'Historical {parameter} Data',
        width=700,
        height=400
    ).interactive()
    
    return chart

def generate_water_quality_report(parameter, forecasted_values, forecast_dates, historical_data):
    """Generate a water quality report without using Gemini API"""
    historical_stats = f"""
    Historical Data Statistics for {parameter}:
    - Mean: {historical_data[parameter].mean():.4f}
    - Standard Deviation: {historical_data[parameter].std():.4f}
    - Minimum: {historical_data[parameter].min():.4f}
    - Maximum: {historical_data[parameter].max():.4f}
    """
    
    forecast_details = "\n".join([
        f"Date: {date.strftime('%Y-%m-%d')}, Predicted Value: {value:.4f}"
        for date, value in zip(forecast_dates, forecasted_values)
    ])
    
    report = f"""
    ## Water Quality Report for {parameter}
    
    ### Historical Context
    {historical_stats}
    
    ### Forecast Summary
    {forecast_details}
    
    ### Analysis
    Based on the forecasted values:
    - Values within ±1 standard deviation from the mean are considered normal
    - Values below -1 standard deviation may indicate improved conditions
    - Values above +1 standard deviation may indicate potential concerns
    
    ### Recommendations
    - Monitor the parameter closely if values exceed normal ranges
    - Compare with regulatory standards for water quality
    - Consider environmental factors that may influence these measurements
    """
    
    return report

def get_status_details(value, parameter_data, selected_parameter):
    """Determine status based on actual parameter thresholds"""
    mean = parameter_data[selected_parameter].mean()
    std = parameter_data[selected_parameter].std()
    
    if value < mean - std:
        return "Low Risk", "green", 30
    elif mean - std <= value < mean + std:
        return "Moderate Risk", "orange", 60
    else:
        return "High Risk", "red", 90

def create_altair_weather_kpi(location, last_date):
    """Create an Altair-based Weather KPI visualization using dummy data"""
    # Generate dummy weather data for Naini
    weather_metrics = pd.DataFrame([
        {'Metric': 'Temperature', 'Value': np.random.uniform(25, 35), 'Unit': '°C'},
        {'Metric': 'Humidity', 'Value': np.random.uniform(50, 85), 'Unit': '%'},
        {'Metric': 'Wind Speed', 'Value': np.random.uniform(5, 15), 'Unit': 'km/h'},
        {'Metric': 'Precipitation', 'Value': np.random.uniform(0, 8), 'Unit': 'mm/h'}
    ])
    
    color_scale = alt.Color('Metric:N', 
        scale=alt.Scale(
            domain=['Temperature', 'Humidity', 'Wind Speed', 'Precipitation'],
            range=['#FF6B6B', '#4ECDC4', '#2575FC', '#8A4FFF']
        )
    )
    
    base = alt.Chart(weather_metrics).mark_bar(
        cornerRadius=5
    ).encode(
        x=alt.X('Metric:N', 
            title=None, 
            axis=alt.Axis(labelAngle=0, labelPadding=5)
        ),
        y=alt.Y('Value:Q', 
            title=None,
            axis=alt.Axis(grid=False)
        ),
        color=color_scale,
        tooltip=['Metric', 'Value', 'Unit']
    ).properties(
        title={
            'text': f'Weather Overview - {location}',
            'subtitle': f'As of {datetime.now().strftime("%Y-%m-%d %H:%M")} (Simulated Data)',
            'color': '#6a11cb',
            'fontSize': 16,
            'fontWeight': 'bold'
        },
        width=400,
        height=250
    )
    
    text = base.mark_text(
        dy=-10,
        color='black',
        fontWeight='bold'
    ).encode(
        text=alt.Text('Value:Q', format='.1f'),
        color=color_scale
    )
    
    return base + text

def enhanced_dashboard_layout():
    """Create an aesthetically pleasing dashboard layout"""
    st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .dashboard-card {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 16px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 30px rgba(31, 38, 135, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(31, 38, 135, 0.15);
    }
    
    .gradient-text {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def display_water_quality_alerts(forecast_values, parameter_data):
    """Display water quality alerts based on forecast values"""
    def get_risk_level(value, parameter_name):
        mean = parameter_data[parameter_name].mean()
        std = parameter_data[parameter_name].std()
        
        if value < mean - std:
            return 'low'
        elif mean - std <= value < mean + std:
            return 'moderate'
        else:
            return 'high'

    risk_config = {
        'low': {
            'icon': '✅',
            'color': 'green',
            'title': 'Low Risk',
            'description': 'Within expected range'
        },
        'moderate': {
            'icon': '⚠️',
            'color': 'orange',
            'title': 'Moderate Risk',
            'description': 'Approaching threshold limits'
        },
        'high': {
            'icon': '🚨',
            'color': 'red',
            'title': 'High Risk',
            'description': 'Exceeding safe limits'
        }
    }

    risk_parameters = []

    for param_name, value in forecast_values.items():
        risk_level = get_risk_level(value, param_name)
        
        if risk_level != 'low':
            risk_parameters.append({
                'name': param_name,
                'value': value,
                'risk_level': risk_level
            })

    if not risk_parameters:
        st.success("✅ Water Quality: All parameters are within safe limits.")
    else:
        for param in risk_parameters:
            config = risk_config[param['risk_level']]
            
            st.markdown(f"""
            <div style="
                background-color: {config['color']}20; 
                border: 2px solid {config['color']}; 
                border-radius: 10px; 
                padding: 15px; 
                margin-bottom: 10px;
            ">
            <h4 style="color: {config['color']}; margin-top: 0;">
                {config['icon']} {config['title']} - {param['name'].replace('_', ' ').title()}
            </h4>
            <p style="margin-bottom: 5px;">{config['description']}</p>
            <p style="font-weight: bold; color: {config['color']};">
                Forecast Value: {param['value']:.2f}
            </p>
            </div>
            """, unsafe_allow_html=True)

def make_donut(input_response, input_text, parameter, parameter_data, selected_parameter):
    """Create an Altair donut chart for water quality forecast visualization"""
    status, risk_color, risk_percentage = get_status_details(
        input_response, parameter_data, selected_parameter
    )
    
    color_map = {
        'Low Risk': ['#27AE60', '#12783D'],
        'Moderate Risk': ['#F39C12', '#875A12'],
        'High Risk': ['#E74C3C', '#FAF9F6']
    }
    
    chart_color = color_map.get(status, color_map['Moderate Risk'])
    
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100 - risk_percentage, risk_percentage]
    })
    
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })
    
    plot = alt.Chart(source).mark_arc(
        innerRadius=45, 
        cornerRadius=25
    ).encode(
        theta="% value",
        color=alt.Color(
            "Topic:N", 
            scale=alt.Scale(
                domain=[input_text, ''],
                range=chart_color
            ),
            legend=None
        )
    ).properties(width=130, height=130)
    
    text = alt.Chart(source).mark_text(
        align='center', 
        color=chart_color[0], 
        font="Arial", 
        fontSize=16, 
        fontWeight='bold'
    ).encode(
        text=alt.value(f'{input_response:.2f}')
    ).properties(width=130, height=130)
    
    plot_bg = alt.Chart(source_bg).mark_arc(
        innerRadius=45, 
        cornerRadius=20
    ).encode(
        theta="% value",
        color=alt.Color(
            "Topic:N", 
            scale=alt.Scale(
                domain=[input_text, ''],
                range=chart_color
            ),
            legend=None
        )
    ).properties(width=130, height=130)
    
    return plot_bg + plot + text

def main():
    st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    .dashboard-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        padding: 20px;
    }
    .gradient-text {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .donut-container {
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        gap: 20px;
        width: 100%;
    }
    .donut-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    enhanced_dashboard_layout()
    st.markdown('<h1 class="gradient-text">NAINI WATER MONITORING STATION</h1>', unsafe_allow_html=True)

    selected_location = "NAINI"
    
    if selected_location:
        df = pd.read_csv(LOCATIONS[selected_location]['file_path'])
        df['Date'] = df['Date'].apply(parse_date)
        last_date = df['Date'].max()

        col1, col2 = st.columns(2)

        with col1:
            create_greeting_card()
            weather_kpi = create_altair_weather_kpi(selected_location, df['Date'].max())
            st.altair_chart(weather_kpi, use_container_width=True)

        with col2:
            location_data = LOCATIONS[selected_location]
            satellite_map = create_satellite_map(location_data['lat'], location_data['lon'])
            folium_static(satellite_map, width=500, height=500)
        
        parameters = [
            param for param in df.columns 
            if param not in ['Date', 'Temperature', 'Rainfall']
        ]

        formatted_parameters = [
            param.replace('_', ' ').title() for param in parameters
        ]

        tabs = st.tabs(formatted_parameters)
        
        for idx, parameter in enumerate(parameters):
            with tabs[idx]:
                try:
                    last_date = df['Date'].max()
                    weather_forecasts = generate_dummy_weather_forecast(last_date)
                    model = load_model_for_parameter(parameter)

                    if model and weather_forecasts:
                        historical_features, exogenous_features, scaler, last_10_days, temp_scaler, rainfall_scaler = prepare_input_data(
                            df, weather_forecasts, parameter
                        )

                        prediction = model.predict([historical_features, exogenous_features])
                        predicted_values = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
                        forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(predicted_values))]

                        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            past_year_data = df[df['Date'] > (last_date - timedelta(days=365))]
                            st.subheader('Past Year Data')
                            historical_chart = create_altair_historical_plot(past_year_data, parameter)
                            st.altair_chart(historical_chart, use_container_width=True)
                        
                        with col2:
                            st.subheader('10 Day Historical and 5 Day Forecast')
                            forecast_chart = create_altair_forecast_plot(last_10_days, predicted_values, parameter)
                            st.altair_chart(forecast_chart, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)

                        forecast_dict = {parameter: predicted_values[-1]}
                        display_water_quality_alerts(forecast_dict, df)

                        st.subheader("Daily Water Quality Forecast")
                        donut_col, report_col = st.columns(2)

                        with donut_col:
                            st.markdown('''
                            <style>
                            .donut-container {
                                display: flex;
                                flex-direction: row;
                                justify-content: center;
                                align-items: center;
                                gap: 20px;
                                width: 100%;
                            }
                            .donut-item {
                                display: flex;
                                flex-direction: column;
                                align-items: center;
                            }
                            </style>
                            <div class="donut-container">
                            ''', unsafe_allow_html=True)

                            for idx, (date, value) in enumerate(zip(forecast_dates, predicted_values)):
                                donut = make_donut(
                                    input_response=value,  
                                    input_text=date.strftime("%Y-%m-%d"),
                                    parameter=parameter,
                                    parameter_data=df,
                                    selected_parameter=parameter
                                )
                                
                                st.markdown(f'<div class="donut-item">', unsafe_allow_html=True)
                                st.altair_chart(donut, use_container_width=False)
                                st.markdown(f'<p>{date.strftime("%Y-%m-%d")}</p>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)

                            st.markdown('</div>', unsafe_allow_html=True)

                        with report_col:
                            try:
                                report = generate_water_quality_report(
                                    parameter, 
                                    predicted_values, 
                                    forecast_dates, 
                                    df
                                )
                                st.markdown("### Water Quality Analysis")
                                st.markdown(report)
                            except Exception as e:
                                st.error(f"Could not generate insights: {e}")

                except Exception as e:
                    st.error(f"Error processing data for {parameter}: {e}")

if __name__ == "__main__":
    main()