import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Battery Dashboard",
    page_icon="🔋",
    layout="wide"
)

# Load models
risk_model = joblib.load("risk_model.pkl")
soh_model = joblib.load("soh_model.pkl")

# Title
st.markdown("<h1 style='text-align: center;'>🔋 Battery Thermal Management Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("⚙ Battery Inputs")

voltage = st.sidebar.slider("Voltage (V)",3.0,4.5,3.7)
current = st.sidebar.slider("Current (A)",0.0,5.0,2.0)
ambient = st.sidebar.slider("Ambient Temperature (°C)",10,50,25)
cycle = st.sidebar.slider("Battery Cycle",1,1000,50)

# Calculations
power = voltage * current
heat = current**2 * 0.015

features = np.array([[voltage,current,power,heat,ambient]])

soh = soh_model.predict([[voltage,current,power,heat,ambient,cycle]])

temp_pred = voltage*10 + current*2 + ambient*0.1

# Dashboard Metrics
st.subheader("📊 Battery Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("🔌 Voltage", f"{voltage:.2f} V")
col2.metric("⚡ Current", f"{current:.2f} A")
col3.metric("🌡 Temperature", f"{temp_pred:.2f} °C")
col4.metric("🔋 Battery Health", f"{soh[0]*100:.2f} %")

st.markdown("---")

# Performance Section
st.subheader("📈 Battery Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Battery Health")
    st.progress(int(soh[0]*100))

with col2:
    temp_percent = min(int((temp_pred/80)*100),100)
    st.write("Temperature Level")
    st.progress(temp_percent)

with col3:
    power_percent = min(int((power/20)*100),100)
    st.write("Power Usage")
    st.progress(power_percent)

st.markdown("---")

# Prediction Results
st.subheader("Prediction Results")

st.write("🌡 Predicted Temperature:", round(temp_pred,2), "°C")
st.write("🔋 Battery Health:", round(soh[0]*100,2), "%")