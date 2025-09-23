import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the pre-trained model
model = tf.keras.models.load_model("models/ann/salary_ann_model_regression.h5")

# Load the scaler and encoders
with open("models/encoders/scaler_regression.pkl", "rb") as file:
    scaler_regression = pickle.load(file)

with open("models/encoders/label_encoder_gender_regression.pkl", "rb") as file:
    label_encoder_gender_regression = pickle.load(file)

with open("models/encoders/onehot_encoder_geo_regression.pkl", "rb") as file:
    onehot_encoder_geo_regression = pickle.load(file)

# Streamlit app
st.title("Estimated Salary Prediction")

# User input
geography = st.selectbox("Geography", onehot_encoder_geo_regression.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender_regression.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.slider("Credit Score", 350, 850)
exited = st.selectbox("Exited", [0, 1])
tenure = st.slider("Tenure (years)", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender_regression.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited]
})

# OHE "Geography" value
geo_encoded = onehot_encoder_geo_regression.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo_regression.get_feature_names_out(['Geography']))

# Combine with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler_regression.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.write(f"Predcited Estimate Salary: ${predicted_salary:.2f}")
