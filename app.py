import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model and preprocessors
model = load_model('churn_model.h5')
scaler = joblib.load('scaler.save')
le = joblib.load('label_encoder.save')

# Function to preprocess input data for prediction
def preprocess_input(input_data):
    # Encode Gender
    input_data['Gender'] = le.transform([input_data['Gender']])[0]

    # One-hot encode Geography: ['France', 'Germany', 'Spain'] with drop_first=True means we use two columns Geography_Germany, Geography_Spain
    geography_cols = ['Geography_Germany', 'Geography_Spain']
    geo_values = [0, 0]
    if input_data['Geography'] == 'Germany':
        geo_values[0] = 1
    elif input_data['Geography'] == 'Spain':
        geo_values[1] = 1
    # Drop original Geography column
    input_data = input_data.drop('Geography')

    # Prepare feature vector in the correct order
    features = [
        input_data['CreditScore'],
        input_data['Gender'],
        input_data['Age'],
        input_data['Tenure'],
        input_data['Balance'],
        input_data['NumOfProducts'],
        input_data['HasCrCard'],
        input_data['IsActiveMember'],
        input_data['EstimatedSalary'],
        geo_values[0],  # Geography_Germany
        geo_values[1],  # Geography_Spain
    ]

    # Scale features
    features_scaled = scaler.transform([features])
    return features_scaled

# Streamlit UI
st.title('Customer Churn Prediction')

# User inputs
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure (years)', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance', min_value=0.0, value=50000.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

# Create DataFrame of inputs for preprocessing
input_dict = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

if st.button('Predict Churn'):
    processed_input = preprocess_input(pd.Series(input_dict))
    prediction_prob = model.predict(processed_input)[0][0]
    st.write(f'Churn Probability: {prediction_prob:.2%}')
    if prediction_prob > 0.5:
        st.warning('Warning: The customer is likely to churn.')
    else:
        st.success('The customer is likely to stay.')

