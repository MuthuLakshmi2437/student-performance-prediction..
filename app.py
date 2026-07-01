import streamlit as st
import numpy as np
import pickle

# Load model only
model = pickle.load(open("model_pickle", "rb"))

st.set_page_config(page_title="Student Performance Predictor")

st.title("🎓 Student Performance Predictor (Simple)")

# Inputs
hours = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)

score = st.number_input("Previous Score", min_value=0, max_value=100, value=70)

extra = st.selectbox("Extracurricular", ["No", "Yes"])

sleep = st.number_input("Sleep Hours", min_value=0, max_value=24, value=6)

papers = st.number_input("Practice Papers", min_value=0, max_value=20, value=2)

# convert Yes/No → 0/1 (only basic encoding)
extra_val = 1 if extra == "Yes" else 0

# predict
if st.button("Predict"):
    input_data = np.array([[hours, score, extra_val, sleep, papers]])

    prediction = model.predict(input_data)

    st.success(f"🎯 Predicted Score: {prediction[0]:.2f}")
