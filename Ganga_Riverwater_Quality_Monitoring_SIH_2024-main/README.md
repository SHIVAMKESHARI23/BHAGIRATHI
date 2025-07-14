# 🌊 BHAGIRATHI – AI-Based Ganga River Water Quality Intelligence System

> 🎯 **Smart India Hackathon 2024 – Software Edition**  
> 🏛️ Ministry of Jal Shakti | 📄 Problem Statement ID: 1694  
> 🔗 [Live Web App](https://bhagirathi-namamiganga.streamlit.app)

---

## 🧭 Overview

**BHAGIRATHI** is an advanced AI-powered Decision Support System (DSS) designed to monitor and forecast the **real-time water quality** of the Ganga River. It combines satellite data, weather feeds, IoT inputs, and predictive models to assist government authorities and the public in proactively managing river health and ensuring sustainability under the **Namami Gange Mission**.

Developed as part of **Smart India Hackathon 2024**, this system embodies the spirit of technology for social good.

---

## 🚀 Key Features

| 🔹 Feature                    | 🌐 Description |
|-----------------------------|----------------|
| 📍 Interactive River Map     | Visualizes real-time water stations, overlays Bhuvan GIS layers |
| 📊 AI Forecasting Engine     | Predicts pH, BOD, DO, and turbidity using ML models |
| 🌦️ Weather Data Integration  | Fetches temperature, rainfall, and humidity for impact modeling |
| 🛰️ Satellite Imagery Overlay | Integrates Bhuvan WMS themes like land use, drainage, pollution zones |
| 📽️ Awareness Module          | AI-generated videos about cultural and ecological relevance of Ganga |
| ⚠️ Alert System              | Color-coded zones and early warnings based on model forecasts |
| 📥 Download Reports          | Exports local station data and forecasts in tabular form |
| 🧠 DSS Simulation            | Supports scenario-based decision-making for authorities |

---

## 📸 Screenshots

> _Replace with actual images in `/assets` for final submission._

| River Map Interface | Forecasting Dashboard |
|---------------------|------------------------|
| ![Map](assets/map_sample.jpg) | ![Forecast](assets/forecast_sample.jpg) |

---

## 🛠️ Tech Stack

- 🐍 Python 3.9
- 📊 Streamlit (frontend + backend)
- 🌐 Folium + Leaflet.js (maps)
- 📦 Pandas, NumPy, Scikit-learn
- 📈 Plotly (charts)
- 🌦️ Open-Meteo API (weather)
- 🛰️ Bhuvan NRSC WMS (satellite overlays)
- 🧠 LSTM / regression models (forecasting)
- 📡 Simulated IoT data for water parameters

---

## 🔧 Project Setup

```bash
# Clone this repository
git clone https://github.com/sanatan0511/BHAGIRATHI.git
cd BHAGIRATHI

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run HomePage.py
