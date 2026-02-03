import streamlit as st
import torch
import numpy as np
from model import PerformancePredictor
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and prepare scaler again
df = pd.read_csv("student-mat.csv", sep=",")
df = df[['studytime', 'absences', 'G1', 'G2', 'health']]
scaler = StandardScaler()
scaler.fit(df.values)

# Load model
model = PerformancePredictor()
model.load_state_dict(torch.load("model.pth"))
model.eval()

st.title("🎓 Academic Performance Predictor")

st.write("Enter student academic details to predict performance category")

# User inputs
studytime = st.slider("Study Time (1–4)", 1, 4, 2)
absences = st.slider("Number of Absences", 0, 100, 5)
G1 = st.slider("Internal Marks 1 (0–20)", 0, 20, 10)
G2 = st.slider("Internal Marks 2 (0–20)", 0, 20, 10)
health = st.slider("Health Status (1–5)", 1, 5, 3)

if st.button("Predict Performance"):
    input_data = np.array([[studytime, absences, G1, G2, health]])
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)

    labels = {
        3: "Excellent 🌟",
        2: "Good 👍",
        1: "Average 🙂",
        0: "At Risk ⚠️"
    }

    st.success(f"Predicted Academic Performance: **{labels[prediction.item()]}**")
