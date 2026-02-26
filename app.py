import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --------------------------
# Load trained Random Forest model
# --------------------------
rf_model = joblib.load("random_forest_model.pkl")

# --------------------------
# Streamlit UI
# --------------------------
st.title("✈️ Flight Price Predictor")

st.header("Flight Details")
departure_hour = st.number_input("Departure Hour (0-23)", 0, 23, 0, help="Enter hour in 24h format")
departure_min = st.number_input("Departure Minute (0-59)", 0, 59, 0, help="Enter minute")
arrival_hour = st.number_input("Arrival Hour (0-23)", 0, 23, 0, help="Enter hour in 24h format")
arrival_min = st.number_input("Arrival Minute (0-59)", 0, 59, 0, help="Enter minute")
duration_hour = st.number_input("Duration Hours", 0, 24, 0, help="Flight duration hours")
duration_min = st.number_input("Duration Minutes", 0, 59, 0, help="Flight duration minutes")
total_stops = st.number_input("Total Stops", 0, 5, 0, help="Number of stops during flight")

st.subheader("Route Details (Optional)")
# Common airport codes
airports = ["DEL", "BOM", "BLR", "MAA", "HYD", "None"]

route_1 = st.selectbox("Route 1", airports)
route_2 = st.selectbox("Route 2", airports)
route_3 = st.selectbox("Route 3", airports)
route_4 = st.selectbox("Route 4", airports)
route_5 = st.selectbox("Route 5", airports)

# --------------------------
# Prepare input dataframe
# --------------------------
# Convert hours/minutes to total minutes
dep_total_min = departure_hour * 60 + departure_min
arr_total_min = arrival_hour * 60 + arrival_min
duration_total_min = duration_hour * 60 + duration_min

# Cyclical encoding for hours
dep_hour_sin = np.sin(2 * np.pi * departure_hour / 24)
dep_hour_cos = np.cos(2 * np.pi * departure_hour / 24)
arr_hour_sin = np.sin(2 * np.pi * arrival_hour / 24)
arr_hour_cos = np.cos(2 * np.pi * arrival_hour / 24)

input_dict = {
    'Dep_total_min': [dep_total_min],
    'Arr_total_min': [arr_total_min],
    'Duration_total_min': [duration_total_min],
    'Dep_hour_sin': [dep_hour_sin],
    'Dep_hour_cos': [dep_hour_cos],
    'Arr_hour_sin': [arr_hour_sin],
    'Arr_hour_cos': [arr_hour_cos],
    'Total_Stops': [total_stops],
    'Route_1': [route_1],
    'Route_2': [route_2],
    'Route_3': [route_3],
    'Route_4': [route_4],
    'Route_5': [route_5]
}

input_df = pd.DataFrame(input_dict)

# One-hot encode route columns
input_df = pd.get_dummies(input_df)

# Align columns with model
model_features = rf_model.feature_names_in_
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]

# --------------------------
# Predict
# --------------------------
if st.button("Predict Price"):
    prediction = rf_model.predict(input_df)
    st.success(f"💰 Predicted Flight Price: ₹{prediction[0]:.2f}")