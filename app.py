import streamlit as st
import numpy as np
import pickle

# Load model
with open("model_pickle", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("model_pickle", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction App")
st.write("Enter student details to predict performance index")

# Inputs
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)

previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=70)

extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])

sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24, value=6)

practice_papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=20, value=2)

# Convert categorical to numeric
extra = 1 if extracurricular == "Yes" else 0

# Prediction button
if st.button("Predict Performance"):
    input_data = np.array([[hours_studied, previous_scores, extra, sleep_hours, practice_papers]])

    # scale input
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"🎯 Predicted Performance Index: {prediction[0]:.2f}")
