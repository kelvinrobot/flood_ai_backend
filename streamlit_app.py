import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import pi

st.set_page_config(page_title="Flood Detection Dashboard", layout="wide")
st.title("⛈️ Real-time Flood Prediction and Explainability Dashboard")

st.markdown("""
This dashboard helps visualize how the AI models detect flood risks using environmental and sensor data.
Below you can:
- Input sensor readings manually
- Send data to the FastAPI backend
- Get flood prediction and severity
- See visual confidence scores and insights using advanced plots
""")

st.sidebar.header("Input Sensor Readings")

with st.sidebar.form("flood_form"):
    avg_temp = st.number_input("Average Temperature (°C)", value=28.5)
    humidity = st.number_input("Humidity (%)", value=76.2)
    precip = st.number_input("Precipitation (mm)", value=12.3)
    windspeed = st.number_input("Wind Speed (km/h)", value=8.1)
    sealevelpressure = st.number_input("Sea Level Pressure (hPa)", value=1012.5)
    cloudcover = st.slider("Cloud Cover (%)", 0, 100, value=68)
    solarradiation = st.number_input("Solar Radiation (W/m2)", value=140.5)
    severerisk = st.slider("Severe Risk Level", 0.0, 1.0, value=0.2)

    st.markdown("**Flood Lag History**")
    flood_lag_1 = st.selectbox("Yesterday Flood?", [0, 1])
    flood_lag_2 = st.selectbox("2 Days Ago?", [0, 1])
    flood_lag_3 = st.selectbox("3 Days Ago?", [0, 1])
    flood_lag_4 = st.selectbox("4 Days Ago?", [0, 1])
    flood_lag_5 = st.selectbox("5 Days Ago?", [0, 1])

    smi_linear_norm = st.slider("Soil Moisture Index (normalized)", 0.0, 1.0, 0.53)
    month = st.selectbox("Month", list(range(1, 13)), index=6)

    submit = st.form_submit_button("Predict Flood")

if submit:
    payload = {
        "Average_temp": avg_temp,
        "humidity": humidity,
        "precip": precip,
        "windspeed": windspeed,
        "sealevelpressure": sealevelpressure,
        "cloudcover": cloudcover,
        "solarradiation": solarradiation,
        "severerisk": severerisk,
        "flood_lag_1": flood_lag_1,
        "flood_lag_2": flood_lag_2,
        "flood_lag_3": flood_lag_3,
        "flood_lag_4": flood_lag_4,
        "flood_lag_5": flood_lag_5,
        "SMI_linear_norm": smi_linear_norm,
        "month": month
    }

    response = requests.post("https://flood-ai-backend-3.onrender.com/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success("Prediction received!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Flood Probability (%)", result["flood_probability_percent"])
            st.metric("Risk Score (%)", result["flood_risk_score_percent"])
            st.metric("Severity Class", result["severity_class"].upper())
            st.metric("Final Flood?", "YES" if result["final_flood"] else "NO")

        with col2:
            st.subheader("Votes From Each Model")
            for vote in result["model_votes"]:
                st.write(f"- {vote}")

       

        # Donut Chart
        st.subheader(" Flood Probability Donut")
        fig, ax = plt.subplots(figsize=(4, 4))
        prob = result["flood_probability_percent"]
        ax.pie([prob, 100 - prob], labels=["Flood", "Safe"], colors=["#FF4B4B", "#3DDC84"],
               startangle=90, wedgeprops={"width": 0.4})
        ax.set(aspect="equal", title="Flood Likelihood")
        st.pyplot(fig)

        # Gauge Chart
        st.subheader(" Flood Risk Gauge")
        risk_score = result["flood_risk_score_percent"]
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh([""], [risk_score], color="crimson")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Flood Risk Score (%)")
        ax.axvline(30, color='green', linestyle='--', label='Low')
        ax.axvline(60, color='orange', linestyle='--', label='Mid')
        ax.axvline(90, color='red', linestyle='--', label='High')
        ax.legend()
        st.pyplot(fig)

        # Flood Probability Plot 
        st.subheader(" Flood Probability Classification")
        prob = result["flood_probability_percent"]
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Flood Probability"], [prob], color="#FF4B4B")
        ax.axvline(50, color='orange', linestyle='--', label='Threshold')
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.legend()
        st.pyplot(fig)

        #  Severity Histogram 
        st.subheader(" Severity Histogram")
        risk_score = result["flood_risk_score_percent"]
        fig, ax = plt.subplots(figsize=(5, 3))
        bins = [0, 30, 60, 100]
        labels = ['Low', 'Mid', 'High']
        severity_zone = pd.cut([risk_score], bins=bins, labels=labels)
        sns.histplot([risk_score], bins=30, kde=False, ax=ax, color='skyblue')
        ax.axvline(risk_score, color='red', linestyle='--')
        ax.set_title(f"Risk Severity Zone: {severity_zone[0]}")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Risk Score (%)")
        st.pyplot(fig)

 

         # Radar Chart
        st.subheader(" Model Confidence Radar")
        radar_labels = ["CatBoost", "LSTM", "XGBoost"]
        radar_values = [1 if "CatBoost: yes" in result["model_votes"] else 0,
                        1 if "LSTM: yes" in result["model_votes"] else 0,
                        result["flood_risk_score_percent"] / 100]
        radar_values += radar_values[:1]
        fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'polar': True})
        angles = [n / float(len(radar_labels)) * 2 * pi for n in range(len(radar_labels))] + [0]
        ax.plot(angles, radar_values, linewidth=1, linestyle='solid')
        ax.fill(angles, radar_values, 'skyblue', alpha=0.4)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_labels)
        st.pyplot(fig)

        # Bar Plot
        st.subheader(" Sensor Readings Overview")
        features = {
            "Average Temp": avg_temp,
            "Humidity": humidity,
            "Precip": precip,
            "Windspeed": windspeed,
            "Sea Pressure": sealevelpressure,
            "Cloud Cover": cloudcover,
            "Solar Radiation": solarradiation,
            "Severe Risk": severerisk,
            "Lag 1": flood_lag_1,
            "Lag 2": flood_lag_2,
            "Lag 3": flood_lag_3,
            "Lag 4": flood_lag_4,
            "Lag 5": flood_lag_5,
            "Soil Moisture": smi_linear_norm
        }
        feat_df = pd.DataFrame(features.items(), columns=["Feature", "Value"])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Value", y="Feature", data=feat_df, palette="coolwarm", ax=ax)
        ax.set_title("Current Sensor Readings")
        st.pyplot(fig)

    else:
        st.error("Prediction failed. Make sure FastAPI backend is running on port 8000.")
