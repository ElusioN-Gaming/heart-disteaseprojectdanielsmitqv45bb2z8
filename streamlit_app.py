import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('best_heart_disease_model.pkl')

# Page configuration
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# App title and description
st.title("Heart Disease Prediction Application")
st.write("""
    This application uses a machine learning model to predict the likelihood of heart disease based on user input. 
    Fill in the details below and click 'Predict' to see the results.
""")

# Function to create a sidebar for user input
def user_input_features():
    st.sidebar.header('User Input Parameters')
    
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male") 
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], 
                              format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (chol)", min_value=0, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], 
                               format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2], 
                                   format_func=lambda x: ["Normal", "Having ST-T Wave Abnormality", "Showing Probable/Definite Left Ventricular Hypertrophy"][x])
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=250, value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=[0, 1], 
                                 format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise Relative to Rest (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2], 
                                 format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.sidebar.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], 
                                format_func=lambda x: ["Normal", "Fixed Defect", "Reversable Defect"][x])
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_data = user_input_features()

# Main panel
st.header('Specified Input parameters')
st.write(input_data)

# Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        st.subheader('Prediction')
        st.write('Heart Disease' if prediction[0] == 1 else 'No Heart Disease')
        st.subheader('Prediction Probability')
        st.write(f"Probability of Heart Disease: {prediction_proba[0][1]:.2f}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Footer
st.sidebar.info("""
    This application was built using a machine learning model trained on heart disease data.
    The model predicts the likelihood of heart disease based on the input parameters provided.
""")
