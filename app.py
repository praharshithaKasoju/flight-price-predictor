import streamlit as st
import pandas as pd
import joblib

# --------------------------
# 1️⃣ Load saved Random Forest model
# --------------------------
rf_model = joblib.load("random_forest_model.pkl")

# --------------------------
# 2️⃣ Streamlit UI
# --------------------------
st.title("✈️ Flight Price Predictor")

st.header("Flight Details")
departure_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23, value=10)
departure_min = st.number_input("Departure Minute (0-59)", min_value=0, max_value=59, value=30)
arrival_hour = st.number_input("Arrival Hour (0-23)", min_value=0, max_value=23, value=12)
arrival_min = st.number_input("Arrival Minute (0-59)", min_value=0, max_value=59, value=45)
duration_hour = st.number_input("Duration Hours", min_value=0, max_value=24, value=2)
duration_min = st.number_input("Duration Minutes", min_value=0, max_value=59, value=15)
total_stops = st.number_input("Total Stops", min_value=0, max_value=5, value=1)

st.subheader("Route Details (Optional)")
route_1 = st.text_input("Route 1", "None")
route_2 = st.text_input("Route 2", "None")
route_3 = st.text_input("Route 3", "None")
route_4 = st.text_input("Route 4", "None")
route_5 = st.text_input("Route 5", "None")

# --------------------------
# 3️⃣ Prepare input dataframe
# --------------------------
input_dict = {
    'Departure_hour': [departure_hour],
    'Departure_min': [departure_min],
    'Arrival_hour': [arrival_hour],
    'Arrival_min': [arrival_min],
    'Duration_hour': [duration_hour],
    'Duration_min': [duration_min],
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

# --------------------------
# 4️⃣ Align columns with model (fill missing ones)
# --------------------------
# Get feature names from trained Random Forest
model_features = rf_model.feature_names_in_  # scikit-learn ≥1.0
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match model
input_df = input_df[model_features]

# --------------------------
# 5️⃣ Predict
# --------------------------
if st.button("Predict Price"):
    prediction = rf_model.predict(input_df)
    st.success(f"💰 Predicted Flight Price: ₹{prediction[0]:.2f}")
