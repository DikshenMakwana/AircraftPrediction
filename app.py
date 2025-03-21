import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Function to load the saved LSTM model
def load_lstm_model():
    model = load_model('lstm_model.h5')  # Load the trained LSTM model
    return model

# Function to preprocess the input data for LSTM
def preprocess_lstm_input(input_data):
    # Assuming the input is a DataFrame with the same features as the model was trained on
    # Scale the input data using MinMaxScaler
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    input_data_scaled = input_data_scaled.reshape((input_data_scaled.shape[0], 1, input_data_scaled.shape[1]))  # Reshaping for LSTM
    return input_data_scaled

# Streamlit app UI
st.title('Aircraft RUL Prediction using LSTM')

st.write("""
    This app predicts the Remaining Useful Life (RUL) of an aircraft based on sensor data using an LSTM model.
""")

# User input form
with st.form(key='input_form'):
    # Collect user input data (same features as used in training the model)
    setting1 = st.number_input("Setting1", min_value=0, max_value=1000, step=1)
    setting2 = st.number_input("Setting2", min_value=0, max_value=1000, step=1)
    setting3 = st.number_input("Setting3", min_value=0, max_value=1000, step=1)
    fan_inlet_temp = st.number_input("Fan inlet temperature (◦R)", min_value=-200, max_value=1000, step=1)
    lpc_outlet_temp = st.number_input("LPC outlet temperature (◦R)", min_value=-200, max_value=1000, step=1)
    hpc_outlet_temp = st.number_input("HPC outlet temperature (◦R)", min_value=-200, max_value=1000, step=1)
    lpt_outlet_temp = st.number_input("LPT outlet temperature (◦R)", min_value=-200, max_value=1000, step=1)
    fan_inlet_pressure = st.number_input("Fan inlet Pressure (psia)", min_value=0, max_value=1000, step=1)
    bypass_duct_pressure = st.number_input("Bypass-duct pressure (psia)", min_value=0, max_value=1000, step=1)
    hpc_outlet_pressure = st.number_input("HPC outlet pressure (psia)", min_value=0, max_value=1000, step=1)
    physical_fan_speed = st.number_input("Physical fan speed (rpm)", min_value=0, max_value=10000, step=1)
    physical_core_speed = st.number_input("Physical core speed (rpm)", min_value=0, max_value=10000, step=1)
    engine_pressure_ratio = st.number_input("Engine pressure ratio (P50/P2)", min_value=0, max_value=100, step=1)
    hpc_outlet_static_pressure = st.number_input("HPC outlet Static pressure (psia)", min_value=0, max_value=1000, step=1)
    fuel_flow_ratio = st.number_input("Ratio of fuel flow to Ps30 (pps/psia)", min_value=0, max_value=100, step=1)
    corrected_fan_speed = st.number_input("Corrected fan speed (rpm)", min_value=0, max_value=10000, step=1)
    corrected_core_speed = st.number_input("Corrected core speed (rpm)", min_value=0, max_value=10000, step=1)
    bypass_ratio = st.number_input("Bypass Ratio", min_value=0, max_value=100, step=1)
    fuel_air_ratio = st.number_input("Burner fuel-air ratio", min_value=0, max_value=100, step=1)
    bleed_enthalpy = st.number_input("Bleed Enthalpy", min_value=0, max_value=1000, step=1)
    required_fan_speed = st.number_input("Required fan speed", min_value=0, max_value=10000, step=1)
    required_fan_conversion_speed = st.number_input("Required fan conversion speed", min_value=0, max_value=10000, step=1)
    high_pressure_turbines_cool_air_flow = st.number_input("High-pressure turbines Cool air flow", min_value=0, max_value=1000, step=1)
    low_pressure_turbines_cool_air_flow = st.number_input("Low-pressure turbines Cool air flow", min_value=0, max_value=1000, step=1)
    failure_within_w1 = st.number_input("Failure within W1", min_value=0, max_value=1000, step=1)
    cycle_norm = st.number_input("Cycle normalization value", min_value=0, max_value=1000, step=1)

    submit_button = st.form_submit_button(label='Predict RUL')

# On submit, make the prediction
if submit_button:
    # Collect the user input into a DataFrame for prediction
    input_data = pd.DataFrame([[
        setting1, setting2, setting3,
        fan_inlet_temp, lpc_outlet_temp, hpc_outlet_temp, lpt_outlet_temp,
        fan_inlet_pressure, bypass_duct_pressure, hpc_outlet_pressure,
        physical_fan_speed, physical_core_speed, engine_pressure_ratio,
        hpc_outlet_static_pressure, fuel_flow_ratio, corrected_fan_speed,
        corrected_core_speed, bypass_ratio, fuel_air_ratio, bleed_enthalpy,
        required_fan_speed, required_fan_conversion_speed,
        high_pressure_turbines_cool_air_flow, low_pressure_turbines_cool_air_flow, 
        failure_within_w1, cycle_norm
    ]], columns=[
        'setting1', 'setting2', 'setting3', 'fan_inlet_temp',
        'lpc_outlet_temp', 'hpc_outlet_temp', 'lpt_outlet_temp',
        'fan_inlet_pressure', 'bypass_duct_pressure', 'hpc_outlet_pressure',
        'physical_fan_speed', 'physical_core_speed', 'engine_pressure_ratio',
        'hpc_outlet_static_pressure', 'fuel_flow_ratio', 'corrected_fan_speed',
        'corrected_core_speed', 'bypass_ratio', 'fuel_air_ratio', 'bleed_enthalpy',
        'required_fan_speed', 'required_fan_conversion_speed', 'high_pressure_turbines_cool_air_flow',
        'low_pressure_turbines_cool_air_flow', 'failure_within_w1', 'cycle_norm'
    ])
    
    # Preprocess the data for LSTM
    X_input_lstm = preprocess_lstm_input(input_data)

    # Load the LSTM model
    model_lstm = load_lstm_model()

    # Make RUL prediction
    rul_prediction = model_lstm.predict(X_input_lstm)

    # Display the predicted RUL
    st.write(f"Predicted Remaining Useful Life (RUL): {rul_prediction[0][0]:.2f} cycles")
