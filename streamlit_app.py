import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('best_heart_disease_model.pkl')

# Define the input features
st.title("Heart Disease Prediction")
st.write("Enter the details of the patient to predict the likelihood of heart disease:")

age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
sex = st.selectbox("Sex", [0, 1])  # 0 for Female, 1 for Male
cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120, step=1)
chol = st.number_input("Serum Cholesterol (chol)", min_value=0, max_value=600, value=200, step=1)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=220, value=150, step=1)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0, step=1)
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Collect input data into a DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Function to make predictions
def make_prediction(data):
    try:
        prediction = model.predict(data)
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Add a button to submit the details
if st.button("Predict"):
    result = make_prediction(input_data)
    if result is not None:
        if result == 1:
            st.success("The patient is likely to suffer from heart disease.")
        else:
            st.success("The patient is unlikely to suffer from heart disease.")

# Documentation and error handling
st.write("## Documentation")
st.write("""
- **Age**: Age of the patient.
- **Sex**: Gender of the patient (0 for Female, 1 for Male).
- **Chest Pain Type (CP)**: Type of chest pain experienced.
- **Resting Blood Pressure (trestbps)**: Resting blood pressure in mm Hg.
- **Serum Cholesterol (chol)**: Serum cholesterol in mg/dl.
- **Fasting Blood Sugar (fbs)**: Fasting blood sugar > 120 mg/dl (0 for False, 1 for True).
- **Resting Electrocardiographic Results (restecg)**: ECG results.
- **Maximum Heart Rate Achieved (thalach)**: Maximum heart rate achieved.
- **Exercise Induced Angina (exang)**: Exercise-induced angina (0 for No, 1 for Yes).
- **ST Depression Induced by Exercise (oldpeak)**: ST depression induced by exercise.
- **Slope of the Peak Exercise ST Segment (slope)**: Slope of the peak exercise ST segment.
- **Number of Major Vessels (ca)**: Number of major vessels colored by fluoroscopy.
- **Thalassemia (thal)**: Thalassemia (0, 1, 2, 3).
""")
