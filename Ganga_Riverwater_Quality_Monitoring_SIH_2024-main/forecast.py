import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import pandas as pd
from bs4 import BeautifulSoup
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression





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
