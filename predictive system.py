# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 17:48:37 2025

@author: arish
"""

import numpy as np
import joblib
import sklearn
from xgboost import XGBRegressor

loaded_model = joblib.load(open('D:\ml deploy\car price/trained_model.pkl'))

input_data = (2014 , 5.59 ,27000,0,0,0,0)
input_data_as_numpy_array = np.array(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)

prediction = loaded_model.predict(input_data_reshaped)
prediction_price_lakhs = prediction[0]
prediction_price_rupees = prediction_price_lakhs * 100000
print(f'Selling price is ${prediction_price_rupees:,.0f}')