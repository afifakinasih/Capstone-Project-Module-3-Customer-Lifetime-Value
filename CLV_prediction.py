import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('model_gradient_boosting_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# Feature options
vehicle_classes = [
    'Luxury Car', 'Luxury SUV', 'SUV', 'Sports Car', 'Two-Door Car', 'Four-Door Car'
]
employment_statuses = [
    'Employed', 'Unemployed', 'Medical Leave', 'Retired'
]
marital_statuses = [
    'Single', 'Married', 'Divorced'
]
renew_offer_types = [
    'Offer1', 'Offer2', 'Offer3', 'Offer4'
]
coverages = [
    'Basic', 'Extended', 'Premium'
]
educations = [
    'High School or Below', 'College', 'Bachelor', 'Master', 'Doctor'
]

st.title("Customer Lifetime Value (CLV) Prediction")

st.write("""
Masukkan data pelanggan untuk memprediksi Customer Lifetime Value (CLV) menggunakan model Gradient Boosting yang telah di-tuning.
""")

with st.form("clv_form"):
    col1, col2 = st.columns(2)
    with col1:
        vehicle_class = st.selectbox("Vehicle Class", vehicle_classes)
        employment_status = st.selectbox("Employment Status", employment_statuses)
        marital_status = st.selectbox("Marital Status", marital_statuses)
        renew_offer_type = st.selectbox("Renew Offer Type", renew_offer_types)
        coverage = st.selectbox("Coverage", coverages)
        education = st.selectbox("Education", educations)
    with col2:
        monthly_premium_auto = st.number_input("Monthly Premium Auto", min_value=62.0, max_value=298.0, value=100.0)
        total_claim_amount = st.number_input("Total Claim Amount", min_value=0.0, max_value=734.69, value=200.0)
        income = st.number_input("Income", min_value=0.0, max_value=99981.0, value=20000.0)
        number_of_policies = st.number_input("Number of Policies", min_value=1, max_value=9, value=2)

    submitted = st.form_submit_button("Predict CLV")

if submitted:
    # Prepare input as DataFrame
    input_dict = {
        'Vehicle Class': [vehicle_class],
        'EmploymentStatus': [employment_status],
        'Marital Status': [marital_status],
        'Renew Offer Type': [renew_offer_type],
        'Coverage': [coverage],
        'Education': [education],
        'Monthly Premium Auto': [monthly_premium_auto],
        'Total Claim Amount': [total_claim_amount],
        'Income': [income],
        'Number of Policies': [number_of_policies]
    }
    input_df = pd.DataFrame(input_dict)

    # Drop unused columns if any
    used_cols = [
        'Vehicle Class', 'EmploymentStatus', 'Marital Status', 'Renew Offer Type',
        'Coverage', 'Education', 'Monthly Premium Auto', 'Total Claim Amount',
        'Income', 'Number of Policies'
    ]
    input_df = input_df[used_cols]

    # Predict
    pred = model.predict(input_df)[0]
    st.success(f"Predicted Customer Lifetime Value (CLV): ${pred:,.2f}")