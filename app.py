import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Global Airline Revenue Engine",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------------------------
# GLOBAL STYLING
# ---------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
}

.main-title {
    font-size: 52px;
    font-weight: 700;
    background: linear-gradient(to right, #002244, #004080);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: gray;
    margin-bottom: 30px;
}

.section-card {
    background: #f8f9fb;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

.price-card {
    background: linear-gradient(135deg, #002244, #004080);
    padding: 35px;
    border-radius: 18px;
    color: white;
    text-align: center;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.2);
}

.stButton>button {
    background: linear-gradient(to right, #002244, #004080);
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(to right, #004080, #002244);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MODE DETECTION
# ---------------------------------------------------
query_params = st.query_params
mode = query_params.get("mode", "landing")

# ===================================================
# LANDING PAGE
# ===================================================
if mode == "landing":

    st.markdown("<div class='main-title'>Global Airline Revenue System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Flight Price Prediction App</div>", unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1436491865332-7a61a109cc05",
        use_container_width=True
    )

    st.markdown("###")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Launch Pricing Engine ✈️"):
            st.query_params.update({"mode": "main"})
            st.rerun()

# ===================================================
# MAIN APPLICATION
# ===================================================
elif mode == "main":

    rf_model = joblib.load("random_forest_model.pkl")

    st.markdown("<div class='main-title'>Flight Pricing Intelligence Portal</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Revenue Optimized Airline Booking System</div>", unsafe_allow_html=True)

    # ---------------- Airline ----------------
    airlines = ["IndiGo", "Air India", "SpiceJet", "Vistara", "GoAir"]
    selected_airline = st.selectbox("Select Airline", airlines)

    airline_multiplier_map = {
        "IndiGo": 1.00,
        "Air India": 1.10,
        "SpiceJet": 0.95,
        "Vistara": 1.20,
        "GoAir": 0.90
    }

    airline_multiplier = airline_multiplier_map[selected_airline]

    # ---------------- Route ----------------
    airports = ["DEL", "BOM", "BLR", "HYD", "MAA"]

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        source = st.selectbox("Source Airport", airports)
    with col_r2:
        destination = st.selectbox("Destination Airport", airports)

    if source == destination:
        st.warning("Source and Destination cannot be the same airport.")
        st.stop()

    # ---------------- Flight Inputs ----------------
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        departure_hour = st.number_input("Departure Hour", 0, 23, 10)
        departure_min = st.number_input("Departure Minute", 0, 59, 0)
        duration_hour = st.number_input("Duration Hours", 0, 24, 2)
        duration_min = st.number_input("Duration Minutes", 0, 59, 30)

    with col2:
        arrival_hour = st.number_input("Arrival Hour", 0, 23, 12)
        arrival_min = st.number_input("Arrival Minute", 0, 59, 0)
        total_stops = st.selectbox("Total Stops", [0, 1, 2, 3])
        journey_date = st.date_input("Journey Date")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Feature Engineering ----------------
    dep_total_min = departure_hour * 60 + departure_min
    arr_total_min = arrival_hour * 60 + arrival_min
    duration_total_min = duration_hour * 60 + duration_min

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
        'Airline': [selected_airline],
        'Source': [source],
        'Destination': [destination]
    }

    input_df = pd.DataFrame(input_dict)
    input_df = pd.get_dummies(input_df)

    model_features = rf_model.feature_names_in_

    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_features]

    # ---------------- Prediction ----------------
    if st.button("Search Live Price 🔍"):

        base_price = rf_model.predict(input_df)[0]

        # Dynamic pricing logic
        demand_multiplier = np.random.uniform(0.95, 1.25)
        seats_left = np.random.randint(10, 120)
        seat_multiplier = 1.20 if seats_left < 20 else 1.10 if seats_left < 50 else 1

        today = datetime.today().date()
        days_left = (journey_date - today).days
        last_minute_multiplier = 1.25 if days_left <= 3 else 1.15 if days_left <= 7 else 1

        peak_multiplier = 1.10 if departure_hour in [6,7,8,9,17,18,19,20] else 1
        fluctuation = np.random.normal(0, 0.03)

        final_price = base_price
        final_price *= airline_multiplier
        final_price *= demand_multiplier
        final_price *= seat_multiplier
        final_price *= last_minute_multiplier
        final_price *= peak_multiplier
        final_price *= (1 + fluctuation)

        final_price = max(final_price, base_price * 0.7)
        final_price = min(final_price, base_price * 3)

        st.markdown("###")

        k1, k2, k3 = st.columns(3)
        k1.metric("Seats Left", seats_left)
        k2.metric("Days To Departure", days_left)
        k3.metric("Demand Multiplier", f"{demand_multiplier:.2f}x")

        st.markdown("###")

        st.markdown(f"""
        <div class='price-card'>
            <h2>Final Ticket Price</h2>
            <h1 style='font-size:48px;'>₹ {final_price:,.2f}</h1>
            <p>{selected_airline} | {source} → {destination}</p>
        </div>
        """, unsafe_allow_html=True)

        st.success("Dynamic Pricing Engine Applied Successfully")
