import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model_pickle", "rb"))

st.title("Student Performance Predictor")

hours = st.number_input("Hours Studied")
score = st.number_input("Previous Score")
extra = st.number_input("Extracurricular (0 or 1)")
sleep = st.number_input("Sleep Hours")
papers = st.number_input("Practice Papers")

if st.button("Predict"):
    input_data = np.array([[hours, score, extra, sleep, papers]])

    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)

    st.success(pred[0])
