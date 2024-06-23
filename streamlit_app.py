import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load('best_heart_disease_model.pkl')

# Define the input fields for user input
st.title("Heart Disease Prediction")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0, 1])  # 0: Female, 1: Male
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120)
chol = st.number_input("Serum Cholesterol (chol)", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

# Create a dataframe from the user input
input_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol],
    'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach], 'exang': [exang],
    'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
})

# Make prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
        st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
