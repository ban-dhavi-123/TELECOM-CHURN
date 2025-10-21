#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("best_churn_model.pkl")       # Your trained model
scaler = joblib.load("scaler.pkl")     # Scaler used during training

# Streamlit page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Customer Churn Prediction App")
st.markdown("Enter the customer details on the left panel and click Predict to see the churn result.")

# Sidebar inputs
st.sidebar.header("Customer Details")

def user_input_features():
    voice_messages = st.sidebar.slider("Voice Messages", 0.0, 500.0, 50.0)
    day_mins = st.sidebar.slider("Day Minutes", 0.0, 400.0, 100.0)
    day_charge = st.sidebar.slider("Day Charge", 0.0, 100.0, 20.0)
    eve_mins = st.sidebar.slider("Evening Minutes", 0.0, 400.0, 100.0)
    eve_charge = st.sidebar.slider("Evening Charge", 0.0, 100.0, 20.0)
    customer_calls = st.sidebar.slider("Customer Service Calls", 0.0, 20.0, 1.0)
    voice_plan = st.sidebar.selectbox("Voice Plan (1 = Yes, 0 = No)", [0, 1])
    intl_plan = st.sidebar.selectbox("International Plan (1 = Yes, 0 = No)", [0, 1])

    data = {
        "voice.messages": voice_messages,
        "day.mins": day_mins,
        "day.charge": day_charge,
        "eve.mins": eve_mins,
        "eve.charge": eve_charge,
        "customer.calls": customer_calls,
        "voice.plan": voice_plan,
        "intl.plan": intl_plan
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale all features that the scaler expects
all_features = ["voice.messages", "day.mins", "day.charge", "eve.mins", "eve.charge", 
                "customer.calls", "voice.plan", "intl.plan"]
input_df[all_features] = scaler.transform(input_df[all_features])

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    result = "Customer will **CHURN**" if prediction[0] == 1 else "Customer will **NOT CHURN**"
    st.subheader("Prediction Result:")
    st.write(result)
    st.subheader("Prediction Probability:")
    st.write(f"Churn Probability: {prediction_proba[0][1]*100:.2f}%")
    st.write(f"Not Churn Probability: {prediction_proba[0][0]*100:.2f}%")


