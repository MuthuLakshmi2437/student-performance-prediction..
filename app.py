import streamlit as st
import numpy as np
import pickle

# Load only model
model = pickle.load(open("model_pickle", "rb"))

st.set_page_config(page_title="Student Predictor")

st.title("🎓 Student Performance Predictor (No scaler file)")

# Inputs
hours = st.number_input("Hours Studied")
score = st.number_input("Previous Score")
extra = st.selectbox("Extracurricular", ["No", "Yes"])
sleep = st.number_input("Sleep Hours")
papers = st.number_input("Practice Papers")

extra_val = 1 if extra == "Yes" else 0

# 🔥 MANUAL SCALING VALUES (TRAINING TIME VALUES MUST MATCH)
mean = np.array([5, 60, 0, 6, 3])
std = np.array([2, 20, 0.5, 2, 1])

if st.button("Predict"):

    input_data = np.array([[hours, score, extra_val, sleep, papers]])

    # manual scaling
    input_scaled = (input_data - mean) / std

    prediction = model.predict(input_scaled)

    st.success(f"🎯 Predicted Score: {prediction[0]:.2f}")
