import streamlit as st
import numpy as np
import pickle

# =========================
# LOAD MODEL + SCALER
# =========================
model = pickle.load(open("model_pickle", "rb"))
scaler = pickle.load(open("scaler_pickle", "rb"))

# =========================
# UI
# =========================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor App")
st.write("Enter student details below:")

# =========================
# INPUTS
# =========================
hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, value=5.0)

score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=70.0)

extra = st.selectbox("Extracurricular Activities", ["No", "Yes"])

sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=6.0)

papers = st.number_input("Practice Question Papers", min_value=0, max_value=20, value=2)

# =========================
# ENCODING (ONLY BASIC)
# =========================
extra_val = 1 if extra == "Yes" else 0

# =========================
# PREDICTION
# =========================
if st.button("Predict Performance"):
    
    # IMPORTANT: correct shape (1,5)
    input_data = np.array([[hours, score, extra_val, sleep, papers]])

    # scaling
    input_scaled = scaler.transform(input_data)

    # prediction
    prediction = model.predict(input_scaled)

    st.success(f"🎯 Predicted Performance: {prediction[0]:.2f}")
