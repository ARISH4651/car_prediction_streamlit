# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 18:15:32 2025
Updated: 27 Sep 2025
@author: arish
"""

import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Load model
loaded_model = joblib.load(r"D:\project\ml deploy\car price\trained_model.pkl")

# Prediction function
def car_price(input_data):
    input_data_np = np.array(input_data, dtype=np.float32).reshape(1, -1)
    prediction = loaded_model.predict(input_data_np)
    prediction_price_lakhs = prediction[0]
    prediction_price_rupees = prediction_price_lakhs * 100000
    return prediction_price_lakhs, prediction_price_rupees

# Main function
def main():
    # Custom Page Config
    st.set_page_config(
        page_title="Car Price Predictor",
        page_icon="🚗",
        layout="wide"
    )

    # Custom CSS for professional look
    st.markdown("""
        <style>
        .stApp {
            background-color: #000000;  /* Black background */
            color: #FFFFFF;             /* White text for readability */
            font-family: 'Helvetica', sans-serif;
        }
        .stButton>button {
            background-color: #4B9CD3;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 24px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #357ABD;
        }
        h1, h2 {
            text-align: center;
            color: #2c3e50;
        }
        .stNumberInput>div>input {
            border-radius: 8px;
            padding: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("🚘 AI-Powered Car Selling Price Prediction")
    st.markdown(
        "Enter your car details below to get an **estimated selling price**, market insights, and tips to sell faster!"
    )

    # Two-column layout for input
    col1, col2 = st.columns(2)

    with col1:
        Year = st.selectbox("📅 Year of the Car", list(range(2004, 2018)))
        Present_Price = st.number_input("💵 Present Price (in Lakhs)", min_value=0.0, step=0.1)
        Kms_Driven = st.number_input("📍 Kilometers Driven", min_value=0, step=1000)

    with col2:
        Fuel_Type = st.selectbox("⛽ Fuel Type", [0, 1, 2], format_func=lambda x: ["Petrol", "Diesel", "CNG"][x])
        Seller_Type = st.selectbox("👤 Seller Type", [0, 1], format_func=lambda x: ["Dealer", "Individual"][x])
        Transmission = st.selectbox("⚙️ Transmission", [0, 1], format_func=lambda x: ["Manual", "Automatic"][x])
        Owner = st.number_input("🧑 Previous Owners", min_value=0, step=1)

    st.markdown("---")

    # Prediction button
    if st.button("🔮 Predict Price"):
        price_lakhs, price_rupees = car_price([Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner])
        
        # Show prediction
        st.success(f"💰 Estimated Selling Price: ₹{price_rupees:,.0f} ({price_lakhs:.2f} Lakhs)")

        # Market comparison (dummy example, you can connect to real API)
        st.info(f"💡 Market Average Price for similar cars: ₹{(price_rupees*1.05):,.0f}")
        
        # Depreciation chart (example)
        years = list(range(Year, Year + 6))
        depreciation = [price_lakhs * (0.85 ** i) for i in range(6)]
        plt.figure(figsize=(8,4))
        plt.plot(years, depreciation, marker='o', color='#4B9CD3')
        plt.title("📉 Depreciation Over Next 5 Years")
        plt.xlabel("Year")
        plt.ylabel("Estimated Price (Lakhs)")
        plt.grid(True)
        st.pyplot(plt)

        # Resale tip
        if price_lakhs < Present_Price * 0.7:
            st.warning("⚠️ Tip: Selling soon is recommended to avoid further depreciation.")
        else:
            st.success("✅ Your car has a good resale value!")

    # Optional: Footer
    st.markdown("---")
    st.markdown("Made by **Arish** | AI-Powered Car Price Predictor")

if __name__ == "__main__":
    main()
