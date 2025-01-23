import streamlit as st
import pandas as pd
import joblib
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(current_dir, 'ensemble_model.pkl')
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()  # Stop execution if the model file is missing

st.title("Loan Approval Prediction")

# Input fields for user input
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
cibil_score = st.number_input("CIBIL Score (300-900)", min_value=300, max_value=900, step=1)
loan_interest = st.number_input("Loan Interest Rate (%)", min_value=0.0, step=0.1)
loan_percent_income = st.number_input("Loan Percent of Income (%)", min_value=0.0, step=0.1)

# Categorical inputs
gender = st.selectbox("Gender", ["Men", "Women"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
employee_status = st.selectbox("Employment Status", ["employed", "self employed", "unemployed", "student"])
residence_type = st.selectbox("Residence Type", ["MORTGAGE", "OWN", "RENT"])
loan_purpose = st.selectbox("Loan Purpose", ["Vehicle", "Personal", "Home Renovation", "Education","Medical", "Other"])

# Default values for missing features
default_values = {
    "active_loans": 0,
    "applicant_age": 30,
    "bank_asset_value": 0,
    "commercial_assets_value": 0,
    "employee_status_student": 0,
    "residential_assets_value": 0,
    "luxury_assets_value": 0,
}

# Prepare the input data
input_data = pd.DataFrame({
    "loan_amount": [loan_amount],
    "cibil_score": [cibil_score],
    "loan_interest": [loan_interest],
    "loan_percent_income": [loan_percent_income],
    "gender_Women": [1 if gender == "Women" else 0],
    "marital_status_Married": [1 if marital_status == "Married" else 0],
    "employee_status_self employed": [1 if employee_status == "self employed" else 0],
    "employee_status_unemployed": [1 if employee_status == "unemployed" else 0],
    "employee_status_student": [1 if employee_status == "student" else 0],
    "residence_type_OWN": [1 if residence_type == "OWN" else 0],
    "residence_type_RENT": [1 if residence_type == "RENT" else 0],
    "loan_purpose_Personal": [1 if loan_purpose == "Personal" else 0],
    "loan_purpose_Home Renovation": [1 if loan_purpose == "Home Renovation" else 0],
    "loan_purpose_Education": [1 if loan_purpose == "Education" else 0],
    "loan_purpose_Vehicle": [1 if loan_purpose == "Vehicle" else 0],
}, index=[0])

# Add default values for missing features
for feature, default_value in default_values.items():
    input_data[feature] = default_value

# Align columns with the model's training features
all_features = model.estimators_[0].feature_names_in_  # Extract features used during training
input_data = input_data.reindex(columns=all_features, fill_value=0)  # Align with expected features

# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)
    if prediction[0] == "Approved":
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
