# -*- coding: utf-8 -*- 
import streamlit as st 
import numpy as np 
import joblib 
import matplotlib.pyplot as plt 
import os 
import pandas as pd

# --- Constants and Setup ---

# Define the Modern Blue accent color for styling
ACCENT_COLOR = "#106FA3" 

# Define Mappings for clarity (used in st.selectbox options)
FUEL_TYPE_OPTIONS = {"Petrol": 0, "Diesel": 1, "CNG": 2}
SELLER_TYPE_OPTIONS = {"Dealer": 0, "Individual": 1}
TRANSMISSION_OPTIONS = {"Manual": 0, "Automatic": 1}

# Image Mapping: Using unique, named placeholders for the top 9 cars in your dataset.
CAR_IMAGE_MAPPING = {
    "Honda City": "https://placehold.co/400x300/1e90ff/white?text=Honda+City",
    "Corolla Altis": "https://placehold.co/400x300/228b22/white?text=Toyota+Corolla+Altis",
    "Hyundai Verna": "https://placehold.co/400x300/ff4500/white?text=Hyundai+Verna",
    "Toyota Fortuner": "https://placehold.co/400x300/8a2be2/white?text=Toyota+Fortuner",
    "Honda Brio": "https://placehold.co/400x300/daa520/white?text=Honda+Brio",
    "Maruti Ciaz": "https://placehold.co/400x300/dc143c/white?text=Maruti+Ciaz",
    "Toyota Innova": "https://placehold.co/400x300/00ced1/white?text=Toyota+Innova",
    "Hyundai i20": "https://placehold.co/400x300/ff69b4/white?text=Hyundai+i20",
    "Grand i10": "https://placehold.co/400x300/708090/white?text=Grand+i10"
}

# Model Loading (Adjust path as necessary)
model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl") 
try:
    loaded_model = joblib.load(model_path)
except FileNotFoundError:
    st.warning("Warning: 'trained_model.pkl' not found. Using a dummy model for visualization.")
    class DummyModel:
        def predict(self, input_data):
            present_price = input_data[0][1]
            kms_driven = input_data[0][2]
            return np.array([max(1.0, present_price * 0.7 - kms_driven / 10000)]) 
    loaded_model = DummyModel()


# Prediction function 
def car_price(input_data): 
    try:
        input_data_np = np.array(input_data, dtype=np.float32).reshape(1, -1) 
        prediction = loaded_model.predict(input_data_np) 
        prediction_price_lakhs = prediction[0] 
        prediction_price_rupees = prediction_price_lakhs * 100000 
        return prediction_price_lakhs, prediction_price_rupees 
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
        return 0.0, 0
        
# --- Main Application ---
def main(): 
    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide") 

    if 'page' not in st.session_state: 
        st.session_state.page = 'landing' 

    # Custom CSS for modern/professional look (FINAL FIX FOR SELECTBOX VISIBILITY)
    st.markdown(f""" 
    <style> 
    /* Global Styles */
    .stApp {{background-color: #1E1E1E; color: #FFFFFF;}} 
    
    /* Landing Page Titles */
    .landing-title {{text-align: center; font-size: 52px; font-weight: bold; margin-bottom:0; color: {ACCENT_COLOR};}} 
    .landing-subtitle {{text-align: center; font-size: 24px; margin-top:5px; margin-bottom:50px; color: #CCCCCC;}} 
    
    /* Car Image Grid */
    .car-image-container {{text-align: center; margin-bottom: 20px;}}
    .car-image {{
        border-radius:12px; 
        transition: transform 0.3s, box-shadow 0.3s; 
        object-fit: cover; 
        height: 180px; 
        width: 100%;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }} 
    .car-image:hover {{transform: scale(1.03); box-shadow: 0px 8px 15px rgba(16, 111, 163, 0.6);}} 
    .car-name-label {{font-weight: bold; font-size: 18px; color: #FFFFFF; margin-top: 10px;}}

    /* Buttons */
    .stButton>button {{
        background-color:black;
        color: #FFFFFF;
        font-size: 20px;
        padding: 10px 40px;
        border-radius: 8px;
        border: none;
        margin-top: 30px;
        transition: 0.3s;
    }} 
    .stButton>button:hover {{background-color: #0E5A8A;}}
    
    /* --- INPUT FIELDS STYLING --- */
    /* Target Number Input fields */
    .stNumberInput>div>input {{
        border-radius:8px; 
        padding:10px; 
        background-color:#2F2F2F; 
        color:#FFFFFF;
        border: 1px solid #444444;
    }} 
    
    /* Target Select Box/Dropdown Input */
    .stSelectbox>div>div>div>div {{
        border-radius:8px; 
        
        height:50px !important;
        padding:10px; 
        background-color:#2F2F2F; 
        border: 1px solid #444444;
    }}
    
    /* CRITICAL FIX: Target the selected text value and its immediate container */
    .stSelectbox>div>div>div>div * {{ 
        color: blue !important; /* Force text color to white */
        background-color: #2F2F2F !important; /* Force background of selected text container */
    }}
    
    /* Ensure all labels use the accent color */
    .stLabel>label {{
        color: #FFFFFF !important; /* Keep the label white */
        font-weight: bold;
    }}
    /* --- END INPUT FIELDS STYLING --- */
    
    /* Headings */
    h1, h2, h3 {{color:{ACCENT_COLOR};}} 
    </style> 
    """, unsafe_allow_html=True) 

# ----------------------------------------------------
# ----------------- LANDING PAGE ----------------- 
# ----------------------------------------------------
    if st.session_state.page == 'landing': 
        st.markdown('<div class="landing-title"> AI-Powered Vehicle Valuation</div>', unsafe_allow_html=True) 
        st.markdown('<div class="landing-subtitle">Discover real-time market estimates for popular cars in India.</div>', unsafe_allow_html=True) 

        car_items = list(CAR_IMAGE_MAPPING.items())
        num_cols = 3
        
        # Robust iteration for 3x3 grid
        for i in range(0, len(car_items), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                linear_index = i + j
                
                if linear_index < len(car_items):
                    car_name, car_url = car_items[linear_index]
                    
                    with cols[j]:
                        img_html = f"""
                        <div class="car-image-container">
                            <img src="{car_url}" class="car-image">
                            <div class="car-name-label">{car_name}</div>
                        </div>
                        """
                        st.markdown(img_html, unsafe_allow_html=True) 

        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True) 
        
        if st.button("Start Valuation"): 
            st.session_state.page = 'predict' 
            st.rerun() 

        st.markdown("</div>", unsafe_allow_html=True) 

# ----------------------------------------------------
# ----------------- PREDICTION PAGE (Single Column) ----------------- 
# ----------------------------------------------------
    elif st.session_state.page == 'predict': 
        st.markdown("---") 
        st.title("Price Prediction Inputs") 
        st.markdown("Please provide the following details about the vehicle.") 
        
        current_year = 2024 
        
        # --- 1. General Details ---
        st.subheader("1. General Details")
        
        # Row 1
        cols_gen = st.columns(2)
        with cols_gen[0]:
             Year = st.selectbox(" Year of Manufacture", list(range(current_year, 2003, -1)), key="year", index=0) 
        with cols_gen[1]:
            Present_Price = st.number_input(" Current Showroom Price (in Lakhs)", min_value=0.1, step=0.1, key="present_price") 
        
        # Row 2
        Kms_Driven = st.number_input(" Kilometers Driven", min_value=100, step=1000, key="kms_driven") 
        
        st.markdown("---") 

        # --- 2. Technical Specifications ---
        st.subheader("2. Technical Specifications")
        
        # Row 3 (3 columns for horizontal flow)
        cols_spec = st.columns(3)
        with cols_spec[0]:
            Fuel_Type_Text = st.selectbox(" Fuel Type", list(FUEL_TYPE_OPTIONS.keys()), key="fuel_type_text")
            Fuel_Type = FUEL_TYPE_OPTIONS[Fuel_Type_Text]
        
        with cols_spec[1]:
            Transmission_Text = st.selectbox("Transmission Type", list(TRANSMISSION_OPTIONS.keys()), key="transmission_text")
            Transmission = TRANSMISSION_OPTIONS[Transmission_Text]
        
        with cols_spec[2]:
            Owner = st.number_input(" Number of Previous Owners", min_value=0, step=1, key="owner", format="%d") 
        
        st.markdown("---")
        
        # --- 3. Seller Information ---
        st.subheader("3. Seller Information")
        
        # Row 4
        Seller_Type_Text = st.selectbox(" Seller Type", list(SELLER_TYPE_OPTIONS.keys()), key="seller_type_text")
        Seller_Type = SELLER_TYPE_OPTIONS[Seller_Type_Text]
            
        st.markdown("---") 

        if st.button(" Calculate Estimated Selling Price"): 
            
            # Input data order must match the model's training order
            input_data = [Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner]
            
            price_lakhs, price_rupees = car_price(input_data) 
            
            if price_lakhs > 0:
                st.success(f" Estimated Selling Price: **₹{price_rupees:,.0f}** ({price_lakhs:,.2f} Lakhs INR)") 

                # Depreciation chart styling update
                years = list(range(Year, Year + 6)) 
                depreciation = [price_lakhs * (0.85 ** i) for i in range(6)] 
                
                fig, ax = plt.subplots(figsize=(8,4)) 
                ax.plot(years, depreciation, marker='o', color=ACCENT_COLOR)
                ax.set_title("Estimated Depreciation Over 5 Years", color=ACCENT_COLOR) 
                ax.set_xlabel("Year") 
                ax.set_ylabel("Estimated Price (Lakhs)") 
                ax.grid(True, alpha=0.3, color='#444444')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color(ACCENT_COLOR)
                fig.patch.set_facecolor('#1E1E1E')
                ax.set_facecolor('#1E1E1E')
                
                st.pyplot(fig)
        
        st.markdown("---")
        if st.button("⬅️ Back to Dashboard"):
            st.session_state.page = 'landing'
            st.rerun()

if __name__ == "__main__": 
    main()