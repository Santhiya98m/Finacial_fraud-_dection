import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('frauddetection.pkl')

# Define the Streamlit app
st.title("FINACIAL TRANSACTION FRAUD DETECTION Web App")

# Description of the app
st.write("""
## About
finacial fraud is a form of identity theft that involves an unauthorized taking of another's personal financial information for the purpose of charging purchases to the account or removing funds from it.

**This Streamlit App aims to detect fraudulent finacial transactions based on transaction details such as amount, sender/receiver information, and transaction type.**

**CONTRIBUTERS:** SANTHIYA, JERIN JASHPER, SELVALAKSHMI, AGANIYA

""")
# Input features of the transaction
st.sidebar.header('Input Features of The Transaction')
step = st.sidebar.slider("Number of Hours it took the Transaction to complete", 0, 100, key="step")
amount = st.sidebar.number_input("Amount", min_value=0.0, max_value=110000.0, key="amount")
oldbalanceOrg = st.sidebar.number_input("Old Balance Orig", min_value=0.0, max_value=110000.0, key="oldbalanceOrg")
newbalanceOrig = st.sidebar.number_input("New Balance Orig", min_value=0.0, max_value=110000.0, key="newbalanceOrig")
oldbalanceDest = st.sidebar.number_input("Old Balance Dest", min_value=0.0, max_value=110000.0, key="oldbalanceDest")
newbalanceDest = st.sidebar.number_input("New Balance Dest", min_value=0.0, max_value=110000.0, key="newbalanceDest")
transaction_type = st.sidebar.selectbox("Type of Transfer Made", ["Option 1", "Option 2", "Option 3"], key="transaction_type")
sender_id = st.sidebar.text_input("Input Sender ID", key="sender_id")
receiver_id = st.sidebar.text_input("Input Receiver ID", key="receiver_id")

# Prediction function
def predict(step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type, nameOrig, nameDest):
    # Encoding transaction type (assuming it's already encoded)
    transaction_types = ["Option 1", "Option 2", "Option 3"]
    transaction_type_encoded = [1 if t == transaction_type else 0 for t in transaction_types]

    # Make prediction
    features = np.array([[step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest] + transaction_type_encoded])
    prediction = model.predict(features)
    return "Fraudulent" if prediction == 1 else "Not Fraudulent"
# Detection result
if st.button("Detect Fraud"):
    result = predict(step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type, sender_id, receiver_id)
    st.write("The transaction is predicted as: {result}")
