import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Battery Thermal Management System",
    layout="wide"
)

st.title("🔋 Battery Thermal Management System")

# Load Models Safely
@st.cache_resource
def load_models():
    models = {}
    
    try:
        models['temp'] = joblib.load("temp_model.pkl")
    except:
        models['temp'] = None
        
    try:
        models['soh'] = joblib.load("soh_model.pkl")
    except:
        models['soh'] = None
        
    try:
        models['risk'] = joblib.load("risk_model.pkl")
    except:
        models['risk'] = None
        
    return models

models = load_models()

# Sidebar Inputs
st.sidebar.header("Battery Parameters")

voltage = st.sidebar.slider("Voltage", 3.0, 4.2, 3.7)
current = st.sidebar.slider("Current", 0.5, 5.0, 2.0)
ambient = st.sidebar.slider("Ambient Temperature", 20, 50, 25)
cycle = st.sidebar.slider("Cycle", 1, 1000, 100)

# Feature Engineering
power = voltage * current
heat = current**2 * 0.015

features = np.array([[voltage, current, power, heat, ambient, cycle]])

st.subheader("Prediction Results")

# Temperature Prediction
if models['temp'] is not None:
    temp = models['temp'].predict(features)[0]
else:
    temp = 25 + current*3 + ambient*0.4

st.metric("Battery Temperature (°C)", round(temp,2))

# SOH Prediction
if models['soh'] is not None:
    soh = models['soh'].predict(features)[0]
else:
    soh = 100 - cycle*0.02

st.metric("Battery Health (%)", round(soh,2))

# Risk Prediction
if models['risk'] is not None:
    risk = models['risk'].predict(features)[0]
else:
    risk = 1 if temp > 45 else 0

if risk == 1:
    st.error("⚠️ Thermal Runaway Risk Detected")
else:
    st.success("✅ Battery Operating Safely")

# Charts
st.subheader("Battery Status")

st.progress(min(int(soh),100))

st.write("Temperature:", round(temp,2))
st.write("SOH:", round(soh,2))
