import streamlit as st
import pandas as pd
import joblib
import gzip

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    with gzip.open("pricing_model.pkl.gz", "rb") as f:
        model, scaler, encoders = joblib.load(f)
    return model, scaler, encoders

st.title("🚖 Dynamic Ride Fare Prediction")

model, scaler, encoders = load_model()

# UI Inputs (no dataset needed)
st.sidebar.header("Enter Ride Information")

number_of_riders = st.sidebar.number_input("Number of Riders", min_value=1, max_value=10, value=2)
location_category = st.sidebar.selectbox("Location Category", ["Urban", "Suburban", "Rural"])
customer_loyalty_status = st.sidebar.selectbox("Customer Loyalty Status", ["Regular", "Silver", "Gold"])
number_of_past_rides = st.sidebar.number_input("Number of Past Rides", min_value=0, value=10)
average_ratings = st.sidebar.number_input("Average Rating", min_value=0.0, max_value=5.0, value=4.5)
time_of_booking = st.sidebar.selectbox("Time of Booking", ["Morning", "Afternoon", "Evening", "Night"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Economy","Premium"])
expected_ride_duration = st.sidebar.number_input("Expected Ride Duration (min)", min_value=1, value=20)

# Prepare input
input_data = pd.DataFrame([{
    "Number_of_Riders": number_of_riders,
    "Location_Category": location_category,
    "Customer_Loyalty_Status": customer_loyalty_status,
    "Number_of_Past_Rides": number_of_past_rides,
    "Average_Ratings": average_ratings,
    "Time_of_Booking": time_of_booking,
    "Vehicle_Type": vehicle_type,
    "Expected_Ride_Duration": expected_ride_duration
}])

# Apply label encoding
for col in encoders:
    input_data[col] = encoders[col].transform(input_data[col])

# Scale features
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Fare"):
    prediction = model.predict(input_scaled)
    st.success(f"💰 Estimated Fare: ${prediction[0]:.2f}")
