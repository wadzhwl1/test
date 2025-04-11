import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Streamlit user interface
st.title("Heart Disease Predictor")

# age: numerical input
age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# sex: categorical selection
sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

if st.button("Predict"):
    advice = 'hhahha'

    st.write(advice)
