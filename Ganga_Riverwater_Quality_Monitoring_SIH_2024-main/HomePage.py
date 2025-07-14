import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import heapq
from collections import defaultdict, deque
import numpy as np
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="Ganga Water Quality Dashboard", layout="wide")
st.title("🌊 Ganga River Water Quality Forecasting System")
st.markdown("""
This dashboard provides real-time and simulated insights into **Ganga River water quality, weather trends**, **industrial discharge**, and **satellite-based indicators**.
""")


class RiverNetwork:
    def __init__(self):
        self.graph = defaultdict(list)
        self.build_network()
    
    def build_network(self):
        """Build Ganga river network with distances (km)"""
        connections = [
            ("Gangotri", "Devprayag", 200),
            ("Devprayag", "Rishikesh", 70),
            ("Rishikesh", "Haridwar", 30),
            ("Haridwar", "Kanpur", 650),
            ("Kanpur", "Prayagraj", 300),
            ("Prayagraj", "Varanasi", 150),
            ("Varanasi", "Patna", 250),
            ("Patna", "Bhagalpur", 200),
            ("Bhagalpur", "Kolkata", 400)
        ]
        for u, v, dist in connections:
            self.graph[u].append((v, dist))
    
    def get_downstream_path(self, location):
        """BFS to find path from location to Kolkata"""
        queue = deque([(location, [location])])
        visited = set()
        
        while queue:
            current, path = queue.popleft()
            if current == "Kolkata":
                return path
            for neighbor, _ in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

locations = {
    "Gangotri": {"station_code": "GNG_GTR", "lat": 30.9943, "lon": 78.9398},
    "Devprayag": {"station_code": "GNG_DPR", "lat": 30.1467, "lon": 78.5986},
    "Rishikesh": {"station_code": "GNG_RSK", "lat": 30.0869, "lon": 78.2676},
    "Haridwar": {"station_code": "GNG_HDR", "lat": 29.9457, "lon": 78.1642},
    "Kanpur": {"station_code": "GNG_KNP", "lat": 26.4499, "lon": 80.3319},
    "Prayagraj": {"station_code": "GNG_PRY", "lat": 25.4358, "lon": 81.8463},
    "Varanasi": {"station_code": "GNG_VNS", "lat": 25.3176, "lon": 82.9739},
    "Patna": {"station_code": "GNG_PTN", "lat": 25.5941, "lon": 85.1376},
    "Bhagalpur": {"station_code": "GNG_BGP", "lat": 25.2532, "lon": 87.0500},
    "Kolkata": {"station_code": "GNG_KOL", "lat": 22.5726, "lon": 88.3639}
}

river_network = RiverNetwork()
selected_location = st.sidebar.selectbox("📍 Select Location", list(locations.keys()))
days_history = st.sidebar.slider("🔕️ Days of History", 7, 30, 10)
forecast_days = st.sidebar.slider("🔮 Forecast Days", 1, 5, 3)

lat = locations[selected_location]['lat']
lon = locations[selected_location]['lon']
station_code = locations[selected_location]['station_code']

#weather section
def fetch_open_meteo(lat, lon, days):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    try:
        res = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "Asia/Kolkata"
        })
        if res.status_code == 200:
            data = res.json()
            df = pd.DataFrame({
                "date": data["daily"]["time"],
                "max_temp": data["daily"]["temperature_2m_max"],
                "min_temp": data["daily"]["temperature_2m_min"],
                "precipitation": data["daily"]["precipitation_sum"]
            })
            return df
        else:
            st.warning("⚠️ Open-Meteo API error.")
            return None
    except Exception as e:
        st.error(f"Weather API error: {e}")
        return None

# WATER QUALITY DATA 
def load_sample_water_quality(parameter, days):
    """More realistic water quality patterns for Ganga"""
    dates = pd.date_range(end=datetime.today(), periods=days).strftime("%Y-%m-%d")
    
    
    base_values = {
        "Gangotri": {"DO": 8.5, "BOD": 1.0, "PH": 7.2, "NO3": 0.5},
        "Haridwar": {"DO": 6.8, "BOD": 2.5, "PH": 7.5, "NO3": 0.8},
        "Kanpur": {"DO": 5.2, "BOD": 6.5, "PH": 7.8, "NO3": 1.2},
        "Varanasi": {"DO": 5.5, "BOD": 5.5, "PH": 7.9, "NO3": 1.0},
        "Kolkata": {"DO": 4.8, "BOD": 6.2, "PH": 8.0, "NO3": 1.5}
    }
    
    
    base = base_values.get(selected_location, {}).get(parameter, 
        5.0 if parameter == "DO" else 
        3.0 if parameter == "BOD" else 
        7.5 if parameter == "PH" else 1.0)
    
    values = [round(base * (1 + 0.1 * np.sin(i/3) + 0.05 * np.random.normal()), 2) 
             for i in range(days)]
    
    unit = "mg/L" if parameter in ["DO", "BOD", "NO3"] else ""
    return pd.DataFrame({"date": dates, "value": values, "unit": unit, "parameter": parameter})

#  SATELLITE DATA SECTION
def load_sample_satellite_data(parameter, days):
    dates = pd.date_range(end=datetime.today(), periods=days).strftime("%Y-%m-%d")
    
    # Parameter-specific patterns
    if parameter == "turbidity":
        values = [round(25 + 10 * np.sin(i/4) + 3 * np.random.normal(), 2) for i in range(days)]
    elif parameter == "water_temp":
        values = [round(28 + 5 * np.sin(i/5) + np.random.normal(), 2) for i in range(days)]
    else:
        values = [round(15 + 5 * np.sin(i/3) + 2 * np.random.normal(), 2) for i in range(days)]
    
    return pd.DataFrame({"date": dates, "value": values, "parameter": parameter})

# -INDUSTRIAL DATA WITH PRIORITY QUEUE --
class IndustrialMonitor:
    def __init__(self):
        self.priority_queue = []
        
    def add_industry(self, name, bod, cod, flow):
        severity = (bod - 50) + (cod - 200)/5  
        heapq.heappush(self.priority_queue, (-severity, name, bod, cod, flow))
    
    def get_most_critical(self, n=5):
        return [heapq.heappop(self.priority_queue) for _ in range(min(n, len(self.priority_queue)))]

@st.cache_data(ttl=3600)
def load_industrial_discharge(station_code):
    monitor = IndustrialMonitor()
    monitor.add_industry("Paper Mill", 90, 320, 2.0)
    monitor.add_industry("Textile Plant", 130, 450, 3.2)
    monitor.add_industry("Dye Factory", 110, 390, 2.5)
    monitor.add_industry("Chemical Plant", 180, 520, 1.8)
    monitor.add_industry("Sugar Mill", 75, 280, 4.0)
    
    critical = monitor.get_most_critical(3)
    data = {
        "industry": [x[1] for x in critical],
        "BOD (mg/L)": [x[2] for x in critical],
        "COD (mg/L)": [x[3] for x in critical],
        "Flow (MLD)": [x[4] for x in critical],
        "Severity": [-x[0] for x in critical]
    }
    return pd.DataFrame(data)

# ENHANCED ALERT SYSTEM
def get_alerts(df):
    alerts = []
    for _, row in df.iterrows():
        if row['parameter'] == 'BOD' and row['value'] > 30:
            alerts.append((3, f"🚨 Critical: High BOD ({row['value']} mg/L)"))
        elif row['parameter'] == 'DO' and row['value'] < 3:
            alerts.append((1, f"⚠️ Warning: Low DO ({row['value']} mg/L)"))
        elif row['parameter'] == 'PH' and (row['value'] < 6.5 or row['value'] > 8.5):
            alerts.append((2, f"⚠️ Warning: Unsafe pH ({row['value']})"))
    
    # Sort by priority
    alerts.sort(reverse=True)
    return [alert[1] for alert in alerts]

#  TABS section
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["💧 Water Quality", "🌦️ Weather Data", "🛰️ Satellite Data", "🏭 Industrial Sewage", "🗺️ River Map", "📈 Forecast", "ℹ️ Project Info"])

# =TAB 1: Water Quality ==
with tab1:
    st.subheader(f"💧 Water Quality at {selected_location}")
    
    wq_parameters = {
        "Dissolved Oxygen (DO)": "DO",
        "Biochemical Oxygen Demand (BOD)": "BOD",
        "pH": "PH",
        "Nitrate": "NO3"
    }
    selected_param = st.selectbox("Select Water Quality Parameter", list(wq_parameters.keys()))
    
    # Historical and forecast data
    hist_data = load_sample_water_quality(wq_parameters[selected_param], days_history)
    forecast_data = load_sample_water_quality(wq_parameters[selected_param], forecast_days)
    
    # Combine data section part-1+2
    combined_dates = list(hist_data['date']) + [
        (datetime.strptime(hist_data['date'].iloc[-1], "%Y-%m-%d") + 
        timedelta(days=i+1)).strftime("%Y-%m-%d") 
        for i in range(forecast_days)
    ]
    combined_values = list(hist_data['value']) + list(forecast_data['value'])
    combined_type = ["historical"]*days_history + ["forecast"]*forecast_days
    
    # Ploting i and ii - more on sanatan0511/Bhagirati// unser this github readme section
    fig = px.line(
        x=combined_dates, 
        y=combined_values,
        color=combined_type,
        labels={"x": "Date", "y": f"{selected_param} ({hist_data['unit'].iloc[0]})"},
        title=f"{selected_param} at {selected_location}"
    )
    fig.update_layout(showlegend=True)
    
    # threshold line
    if selected_param == "DO":
        fig.add_hline(y=4, line_dash="dash", line_color="red", annotation_text="Critical Level")
    elif selected_param == "BOD":
        fig.add_hline(y=6, line_dash="dash", line_color="orange", annotation_text="Safety Threshold")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Alerts
    alerts = get_alerts(hist_data)
    if alerts:
        st.error("### ⚠️ Water Quality Alerts")
        for alert in alerts:
            st.error(alert)
    else:
        st.success("✅ Water quality within safe limits")

# ========== TAB 2: Weather Data ==========
with tab2:
    st.subheader(f"🌦️ Weather Conditions at {selected_location}")

    # --- Open-Meteo Data ---
    weather_df = fetch_open_meteo(lat, lon, days_history)
    open_meteo_current = weather_df.iloc[-1] if weather_df is not None else None

    # --- Simulated IMD Mausam Data ---
    @st.cache_data(ttl=86400)
    def fetch_imd_mausam_data():
        return {
            "Gangotri": {"max_temp": 18.0, "min_temp": 9.0, "precipitation": 2.0},
            "Devprayag": {"max_temp": 27.0, "min_temp": 18.0, "precipitation": 4.0},
            "Rishikesh": {"max_temp": 30.0, "min_temp": 21.0, "precipitation": 3.0},
            "Haridwar": {"max_temp": 32.0, "min_temp": 24.0, "precipitation": 3.0},
            "Kanpur": {"max_temp": 35.0, "min_temp": 27.0, "precipitation": 5.0},
            "Prayagraj": {"max_temp": 36.0, "min_temp": 28.0, "precipitation": 6.0},
            "Varanasi": {"max_temp": 34.0, "min_temp": 26.0, "precipitation": 4.0},
            "Patna": {"max_temp": 33.0, "min_temp": 26.0, "precipitation": 5.0},
            "Bhagalpur": {"max_temp": 32.0, "min_temp": 25.0, "precipitation": 6.0},
            "Kolkata": {"max_temp": 33.0, "min_temp": 28.0, "precipitation": 8.0}
        }

    imd_data = fetch_imd_mausam_data().get(selected_location)

    # --- Visual Crossing API Integration ---
    def fetch_visualcrossing_weather(lat, lon):
        api_key = "AABXNMLM2EHQBQRG2FPXQZ42P"  # ✅ ONLY the API key
        try:
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/today?unitGroup=metric&key={api_key}&include=days"
            res = requests.get(url)
            if res.status_code == 200:
                today = res.json()["days"][0]
                return {
                    "max_temp": today["tempmax"],
                    "min_temp": today["tempmin"],
                    "precipitation": today["precip"]
                }
        except Exception as e:
            st.warning(f"❌ Visual Crossing error: {e}")
        return None

    vc_data = fetch_visualcrossing_weather(lat, lon)

    # --- Display Open-Meteo Current Weather ---
    if open_meteo_current is not None:
        cols = st.columns(3)
        cols[0].metric("🌡️ Max Temp", f"{open_meteo_current['max_temp']}°C")
        cols[1].metric("🌡️ Min Temp", f"{open_meteo_current['min_temp']}°C")
        cols[2].metric("🌧️ Precipitation", f"{open_meteo_current['precipitation']} mm")

    st.divider()

    # --- Weather Source Comparison ---
    st.markdown("### 📊 Compare Weather from Multiple Sources")

    def show_source_row(label, data):
        cols[0].write(f"**{label}**")
        if data:
            cols[1].write(f"{data['max_temp']}°C")
            cols[2].write(f"{data['min_temp']}°C")
            cols[3].write(f"{data['precipitation']} mm")
        else:
            cols[1].write("—")
            cols[2].write("—")
            cols[3].write("—")

    cols = st.columns(4)
    cols[0].write("🔍 Source")
    cols[1].write("🌡️ Max Temp")
    cols[2].write("🌡️ Min Temp")
    cols[3].write("🌧️ Precip")

    show_source_row("📡 Open-Meteo", open_meteo_current.to_dict() if open_meteo_current is not None else None)
    show_source_row("🇮🇳 IMD Mausam (Simulated)", imd_data)
    show_source_row("🌐 Visual Crossing", vc_data)

    st.info("✅ Multiple weather sources improve reliability, especially in sensitive regions like Uttarakhand & UP.")

    # --- Weather Trends Charts ---
    if weather_df is not None:
        st.markdown("### 📈 Historical Weather Trends")

        fig1 = px.line(weather_df, x='date', y=['max_temp', 'min_temp'],
                       labels={"value": "Temperature (°C)", "date": "Date"},
                       title="Temperature Trend")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(weather_df, x='date', y='precipitation',
                      labels={"precipitation": "Rainfall (mm)", "date": "Date"},
                      title="Rainfall Pattern")
        st.plotly_chart(fig2, use_container_width=True)


# ========== TAB 3: Satellite Data ==========
# ========== TAB 3: Satellite Observations ==========

with tab3:
    st.subheader(f"🛰️ Satellite Observations near {selected_location}")

    # Satellite parameters from Bhuvan WMS
    bhuvan_layers = {
        "Turbidity (NTU)": "Ganga_Turbidity",
        "Water Temperature (°C)": "Ganga_Water_Temp",
        "Chlorophyll (μg/L)": "Ganga_Chlorophyll",
        "Suspended Solids (mg/L)": "Ganga_Suspended_Sediment"
    }
    
    selected_sat_param = st.selectbox("Select Parameter", list(bhuvan_layers.keys()))
    selected_layer = bhuvan_layers[selected_sat_param]

    st.markdown(f"""
    **📡 Source**: [Bhuvan Ganga Portal](https://bhuvan-app1.nrsc.gov.in/mowr_ganga/)  
    **🌐 Layer**: `{selected_layer}`  
    **📍 Location**: `{selected_location}`  
    """)

    # Optional: keep fallback trend chart
    df_sat = load_sample_satellite_data(selected_layer.lower(), days_history)
    fig = px.line(df_sat, x='date', y='value', title=f"{selected_sat_param} - Trend (Simulated Fallback)")
    st.plotly_chart(fig, use_container_width=True)

    # Render map with real WMS layer from Bhuvan
    st.subheader("🌍 Real Satellite Imagery (Bhuvan WMS)")
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="OpenStreetMap")

    # Real WMS tile
    folium.raster_layers.WmsTileLayer(
        url="https://bhuvan-app1.nrsc.gov.in/bhuvan/wms",
        name=selected_layer,
        layers=f"Bhuvan:{selected_layer}",
        fmt="image/png",
        transparent=True,
        attribution="Bhuvan NRSC",
        version="1.1.1"
    ).add_to(m)

    # Add marker and river path
    folium.CircleMarker(
        location=[lat, lon],
        radius=8,
        color="red",
        fill=True,
        popup=f"{selected_location} Monitoring Station"
    ).add_to(m)

    # Add river path
    path = river_network.get_downstream_path(selected_location)
    if path:
        folium.PolyLine(
            [(locations[loc]['lat'], locations[loc]['lon']) for loc in path],
            color="blue",
            weight=4,
            popup="Ganga River Path"
        ).add_to(m)

    folium.LayerControl().add_to(m)
    folium_static(m, width=1000, height=600)


# ========== TAB 4: Industrial Sewage ==========
with tab4:
    st.subheader(f"🏭 Industrial Discharge Near {selected_location}")
    df_industry = load_industrial_discharge(station_code)
    
    # Display with color coding
    def color_high(val):
        color = 'red' if val > 100 else 'orange' if val > 50 else 'green'
        return f'color: {color}'
    
    st.dataframe(df_industry.style.applymap(color_high, subset=['BOD (mg/L)', 'COD (mg/L)']), 
                use_container_width=True)
    
    # Pollution flow analysis
    st.subheader("Downstream Impact Analysis")
    downstream = river_network.get_downstream_path(selected_location)
    if len(downstream) > 1:
        st.warning(f"⚠️ Pollution from {selected_location} may affect: {', '.join(downstream[1:])}")
    else:
        st.info("✅ No major downstream locations from this point")

# ========== TAB 5: River Map ==========
with tab5:
    st.subheader("🗺️ Ganga River Network")
    
    # Create comprehensive river network map
    m = folium.Map(location=[26.0, 82.0], zoom_start=6)
    
    # Add all locations
    for name, info in locations.items():
        # Get weather for popup
        weather = fetch_open_meteo(info['lat'], info['lon'], 1)
        if weather is not None:
            temp = f"{weather.iloc[0]['max_temp']}°C / {weather.iloc[0]['min_temp']}°C"
        else:
            temp = "Weather data unavailable"
        
        # Different icons for different locations
        icon_color = "red" if name == selected_location else "blue"
        
        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=f"<b>{name}</b><br>Temp: {temp}",
            icon=folium.Icon(color=icon_color)
        ).add_to(m)
    
    # Add river path
    ganga_path = ["Gangotri", "Devprayag", "Rishikesh", "Haridwar", "Kanpur", 
                 "Prayagraj", "Varanasi", "Patna", "Bhagalpur", "Kolkata"]
    folium.PolyLine(
        [(locations[loc]["lat"], locations[loc]["lon"]) for loc in ganga_path],
        color="blue",
        weight=3,
        popup="Ganga Main Stem"
    ).add_to(m)
    
    folium_static(m, width=1000, height=700)
    
    # River network analysis
    st.subheader("River Network Analysis")
    st.write(f"**Current Location:** {selected_location}")
    st.write(f"**Downstream Path:** {' → '.join(river_network.get_downstream_path(selected_location))}")

# ========== TAB 6: Forecast ==========
with tab6:
    st.subheader("📈 Advanced Water Quality Forecasting")
    
    #  original forecast tab content starts here
    st.title("🌊 Ganga River - City Maps + Real-time Environmental Data + ML Forecast")

    city_bbox = {
        "Gangotri": {"bbox": [78.93, 30.98, 78.95, 31.01], "info": "Gangotri: Origin of River Ganga in the Himalayas.", "layer": "sisdp:BM10K_1213_0101"},
        "Devprayag": {"bbox": [78.59, 30.15, 78.62, 30.18], "info": "Devprayag: Confluence of Alaknanda and Bhagirathi.", "layer": "sisdp:BM10K_1213_0211"},
        "Rishikesh": {"bbox": [78.25, 30.05, 78.35, 30.12], "info": "Rishikesh: Yoga capital and Himalayan foothills gateway.", "layer": "sisdp:BM10K_1213_0317"},
        "Haridwar": {"bbox": [78.13, 29.93, 78.20, 30.00], "info": "Haridwar: Where the Ganga enters the plains from the Himalayas.", "layer": "sisdp:BM10K_1213_0415"},
        "Farrukhabad": {"bbox": [79.55, 27.36, 79.65, 27.42], "info": "Farrukhabad: Historic city along the Ganga.", "layer": "sisdp:BM10K_1213_0601"},
        "Kanpur": {"bbox": [80.20, 26.40, 80.50, 26.55], "info": "Kanpur: Major industrial center with Ganga river pollution concerns.", "layer": "sisdp:BM10K_1213_0615"},
        "Prayagraj": {"bbox": [81.75, 25.35, 81.95, 25.50], "info": "Prayagraj: Triveni Sangam - Confluence of Ganga, Yamuna, Saraswati.", "layer": "sisdp:BM10K_1213_0712"},
        "Varanasi": {"bbox": [82.95, 25.25, 83.15, 25.40], "info": "Varanasi: A spiritual and cultural hub on the Ganga banks.", "layer": "sisdp:BM10K_1213_0939"},
        "Buxar": {"bbox": [83.95, 25.57, 84.05, 25.65], "info": "Buxar: Ganga flows through Bihar here.", "layer": "sisdp:BM10K_1213_1002"},
        "Patna": {"bbox": [85.00, 25.55, 85.25, 25.65], "info": "Patna: Historic city along the Ganga with dense urban population.", "layer": "sisdp:BM10K_1213_0809"},
        "Bhagalpur": {"bbox": [86.95, 25.23, 87.05, 25.30], "info": "Bhagalpur: Silk city with riverine biodiversity.", "layer": "sisdp:BM10K_1213_1012"},
        "Malda": {"bbox": [88.12, 25.00, 88.22, 25.10], "info": "Malda: Key city before Ganga enters the delta.", "layer": "sisdp:BM10K_1213_2007"},
        "Murshidabad": {"bbox": [88.25, 24.17, 88.35, 24.27], "info": "Murshidabad: Historic Bengal city on the river.", "layer": "sisdp:BM10K_1213_2009"},
        "Kolkata": {"bbox": [88.25, 22.50, 88.45, 22.65], "info": "Kolkata: Delta region city where the Ganga meets the Bay of Bengal.", "layer": "sisdp:BM10K_1213_2105"},
        "Ganga Sagar": {"bbox": [88.01, 21.61, 88.05, 21.66], "info": "Ganga Sagar: The point where Ganga meets the sea.", "layer": "sisdp:BM10K_1213_2201"}
    }

    # ----------------------- THEMATIC LAYERS -----------------------
    thematic_layers = {
        "Drainage": "nmcg:drainage",
        "Pollution Sources": "nmcg:pollutionsource",
        "Water Quality": "nmcg:waterquality",
        "Utilities": "nmcg:utilities",
        "Treatment Plants": "nmcg:treatmentplants",
        "Monitoring Locations": "nmcg:monitoringlocations",
        "Project Info": "nmcg:projectrelatedinfo",
        "National Parks & Sanctuaries": "nmcg:nationalparks",
        "Ganga Flood Plain": "nmcg:gangafloodplain"
    }

    selected_city = st.selectbox("📍 Select a City along Ganga River:", list(city_bbox.keys()))
    selected_layers = st.multiselect("🧽 Select additional Bhuvan layers to show:", options=thematic_layers.keys())
    st.info(f"ℹ️ {city_bbox[selected_city]['info']}")

    city_bounds = city_bbox[selected_city]["bbox"]
    center_lat = (city_bounds[1] + city_bounds[3]) / 2
    center_lon = (city_bounds[0] + city_bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17)

    folium.Rectangle(bounds=[[city_bounds[1], city_bounds[0]], [city_bounds[3], city_bounds[2]]], color="blue", fill=True, fill_opacity=0.1, tooltip=selected_city).add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri Satellite", name="🌍 Satellite View", overlay=False, control=True).add_to(m)

    wms_url = "https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms"
    folium.raster_layers.WmsTileLayer(url=wms_url, name="Base City Layer", layers=city_bbox[selected_city]["layer"], fmt="image/png", transparent=True, version="1.1.1", attr="Bhuvan NRSC / ISRO", overlay=True, control=True).add_to(m)

    for layer_name in selected_layers:
        folium.raster_layers.WmsTileLayer(url=wms_url, name=layer_name, layers=thematic_layers[layer_name], fmt="image/png", transparent=True, version="1.1.1", attr="Bhuvan NRSC / ISRO", overlay=True, control=True).add_to(m)

    folium.LayerControl().add_to(m)
    folium_static(m)

    # ----------------------- DATA FETCHING SECTION -----------------------
    st.subheader("📊 Explore Ganga Data from Bhuvan (Experimental)")
    layer_for_data = st.selectbox("📂 Select Layer to Try Extracting Data:", list(thematic_layers.keys()))

    if st.button("📅 Try Fetch Data for Selected Layer"):
        bbox = city_bbox[selected_city]['bbox']
        bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        getfeature_url = (
            f"https://bhuvan-vec1.nrsc.gov.in/bhuvan/wms?service=WMS"
            f"&version=1.1.1&request=GetFeatureInfo"
            f"&layers={thematic_layers[layer_for_data]}"
            f"&bbox={bbox_str}&width=512&height=512"
            f"&srs=EPSG:4326&styles=&feature_count=10"
            f"&query_layers={thematic_layers[layer_for_data]}"
            f"&info_format=text/html&x=256&y=256"
        )
        try:
            response = requests.get(getfeature_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                tables = pd.read_html(str(soup))
                if tables:
                    df = tables[0]
                    st.success(f"✅ Retrieved {len(df)} records from '{layer_for_data}' layer.")
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download CSV", csv, f"{layer_for_data}_data.csv", "text/csv")
                else:
                    st.warning("⚠️ No structured data found in this layer.")
            else:
                st.error("❌ Failed to fetch data.")
        except Exception as e:
            st.error("⚠️ Error occurred while fetching or parsing data.")
            st.exception(e)

    # ----------------------- ML INTEGRATION (DEMO) -----------------------
    st.subheader("🤖 ML Forecast - Simulated Water Quality Trend")
    if st.button("Run ML Forecast"):
        np.random.seed(42)
        X = np.arange(10).reshape(-1, 1)
        y = np.array([7.2, 7.0, 6.8, 6.9, 7.1, 6.7, 6.6, 6.8, 7.0, 6.9])
        model = LinearRegression().fit(X, y)
        pred = model.predict(X)
        df_ml = pd.DataFrame({"Day": X.flatten(), "Observed pH": y, "Predicted pH": pred})
        st.line_chart(df_ml.set_index("Day"))

    # ----------------------- OTHER DATA SOURCES -----------------------
    st.subheader("🔗 External Data Sources (Live APIs & Integration)")
    st.markdown("""
    - 🌧️ **IMD Daily Weather**: [IMD Mausam](https://mausam.imd.gov.in/) → Use OpenMeteo/IMD weather APIs for rainfall/temp
    - 🛰️ **Land Use from Bhuvan**: [Bhuvan](https://bhuvan.nrsc.gov.in/home/index.php)
    - 👥 **Census & Population**: [India WRIS](https://indiawris.gov.in/wris/#/home)
    - 🌊 **Flood Discharge**: [NASA Flood Observatory](https://floodobservatory.colorado.edu/DischargeAccess.html)
    - 🧪 **Water Quality**: [CPCB NWMP](https://cpcb.nic.in/nwmp-data/)

    🚀 Data pipelines being integrated: These sources will power real-time ML + DSS in next version.
    """)
    

# ========== TAB 7: Project Info ==========
with tab7:
    st.subheader("ℹ️ About This Project")
    
    st.markdown("""
    ## Smart India Hackathon 2024 Project
    **Project Name:** Ganga River Water Quality Forecasting System  
    **Team Name:**Bhagirathi-NamamiGanga  
    **Problem Statement:** NR116 - Real-time water quality monitoring and prediction for Ganga River  
    **Organization:** National Mission for Clean Ganga (NMCG)  
    
    ### Project Description
    This dashboard is part of our Smart India Hackathon 2023 submission, designed to:
    - Monitor real-time water quality parameters along the Ganga River
    - Predict future water quality using machine learning models
    - Identify pollution sources and their downstream impact
    - Provide actionable insights for authorities and researchers
    
    ### Technology Stack
    - **Backend:** Python, TensorFlow, Scikit-learn
    - **Frontend:** Streamlit, Plotly, Folium
    - **Data Sources:** CPCB, IMD, Sentinel-2 Satellite
    - **Hosting:** AWS EC2
    
    ### YouTube Demonstration
    Watch our project demonstration video:
    """)
    
    # YouTube embed
    st.video("https://youtu.be/UdJ7rrji4lc?si=NR_7HHGogUq5hHWW")
    
    st.markdown("""
    ### Team Members
    1. Sanatan Singh (Leader && GIS  && Backend Developer &&API )
    2. Shivam kesari - Framework & DSA
    3. Nippon Tadrishi- Datascience && visualization && DSA
    4. Shyam Gupta  - Datascience && visualization && DSA
    5. prayas - Environmental &&ppt && DSA && CONTENT && LOGISTICS && building framework 
    
    ### Contact Information
    Email: sanatansingh23@lpu.in 
    Namami Ganga: [https://nmcg.nic.in/NamamiGanga.aspx)  
    """)
