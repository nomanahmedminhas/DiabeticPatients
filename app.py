import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
ann_diabetes_model = tf.keras.models.load_model('ann_diabetes_model.h5')

# Load the encoders and scaler
sc=pickle.load(open('sc_diabetes.pkl','rb'))

## streamlit app
st.title('Pateints Diabetes Prediction')

# User input
Pregnancies = st.slider('Pregnancies', 0, 5)
Glucose = st.number_input('Glucose')
BloodPressure = st.number_input('Blood Pressure')
SkinThickness = st.number_input('Skin Thickness')
Insulin = st.number_input('Insulin')
BMI = st.number_input('BMI')
DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', 0.0, 3.0)
Age = st.slider('Age', 1, 99)


# Prepare the input data
input_data = pd.DataFrame({
    'Pregnancies': [Pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    'BMI': [BMI],
    'DiabetesPedigreeFunction': [DiabetesPedigreeFunction],
    'Age': [Age],
})

# Scale the input data
input_data_scaled = sc.transform(input_data)


# Predict churn
prediction = ann_diabetes_model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Patient Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The Patient is likely to be Diabetic.')
else:
    st.write('The Patient is likely to be Non-Diabetic.')
