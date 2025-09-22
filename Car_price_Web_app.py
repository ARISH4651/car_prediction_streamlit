# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 18:15:32 2025

@author: arish
"""

import numpy as np
import joblib
import streamlit as st

# Load model
loaded_model = joblib.load(open(r"D:\ml deploy\car price\trained_model.pkl", "rb"))

# Prediction function
def car_price(input_data):
    input_data_np = np.array(input_data, dtype=np.float32).reshape(1, -1)
    prediction = loaded_model.predict(input_data_np)
    prediction_price_lakhs = prediction[0]
    prediction_price_rupees = prediction_price_lakhs * 100000
    return f"💰 Estimated Selling Price: ₹{prediction_price_rupees:,.0f}"

# Main function
def main():
    # Custom Page Config
    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7fa;
            padding: 20px;
            border-radius: 15px;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 24px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("🚘 Car Selling Price Prediction")
    st.write("Enter your car details below to get an **AI-powered price estimate**.")

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        Year = st.selectbox("📅 Year of the Car", list(range(2000, 2025)))
        Present_Price = st.number_input("💵 Present Price (in Lakhs)", min_value=0.0, step=0.1)
        Kms_Driven = st.number_input("📍 Kilometers Driven", min_value=0, step=1000)

    with col2:
        Fuel_Type = st.selectbox("⛽ Fuel Type", [0, 1, 2], format_func=lambda x: ["Petrol", "Diesel", "CNG"][x])
        Seller_Type = st.selectbox("👤 Seller Type", [0, 1], format_func=lambda x: ["Dealer", "Individual"][x])
        Transmission = st.selectbox("⚙️ Transmission", [0, 1], format_func=lambda x: ["Manual", "Automatic"][x])
        Owner = st.number_input("🧑 Previous Owners", min_value=0, step=1)

    # Prediction button
    if st.button("🔮 Predict Price"):
        predict = car_price([Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner])
        st.success(predict)

if __name__ == "__main__":
    main()
