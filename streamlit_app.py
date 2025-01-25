import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO
from fpdf import FPDF
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="💸",
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #ffffff, #e6f7ff);
            font-family: 'Arial', sans-serif;
        }
        .header-container {
            background: linear-gradient(to right, #4CAF50, #5ecf5e);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }
        .header-container h1 {
            font-size: 40px;
        }
        .header-container p {
            font-size: 20px;
            margin-top: 5px;
        }
        .loan-section {
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header section
st.markdown(
    """
    <div class="header-container">
        <h1>AI Loan Application System</h1>
        <p>Smart, Reliable, and Transparent Loan Processing</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load model
model_path = 'best_features_model.pkl'
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Sidebar for Loan EMI Calculator
with st.sidebar:
    st.markdown("### Loan EMI Calculator")
    loan_amount_cal = st.number_input("Loan Amount (INR):", min_value=10000, step=1000, value=500000)
    interest_rate = st.number_input("Interest Rate (%):", min_value=1.0, step=0.1, value=7.0)
    tenure_years = st.number_input("Loan Tenure (Years):", min_value=1, step=1, value=5)

    # EMI Calculation
    monthly_rate = interest_rate / (12 * 100)
    tenure_months = tenure_years * 12
    emi = (loan_amount_cal * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)

    st.write(f"**Estimated EMI:** ₹{emi:,.2f}")

# Step-by-Step Loan Workflow
st.markdown("### Loan Application Steps")
step = st.select_slider(
    "Navigate through the steps:",
    options=["Personal Information", "Loan Details", "Upload Documents", "Final Decision"]
)

if step == "Personal Information":
    st.markdown("#### Step 1: Personal Information")
    st.text_input("Full Name")
    st.text_input("Email Address")
    st.text_input("Phone Number")

elif step == "Loan Details":
    st.markdown("#### Step 2: Loan Details")
    cibil_score = st.slider("CIBIL Score (300-900):", min_value=300, max_value=900, step=1, value=750)
    income_annum = st.number_input("Annual Income (INR):", min_value=0, step=10000, value=5000000)
    loan_amount = st.number_input("Loan Amount (INR):", min_value=0, step=10000, value=2000000)
    loan_term = st.number_input("Loan Term (Months):", min_value=1, step=1, value=24)
    loan_percent_income = st.number_input("Loan Percent of Income (%):", min_value=0.0, step=0.1, value=20.0)
    active_loans = st.number_input("Number of Active Loans:", min_value=0, step=1, value=1)

    gender = st.selectbox("Gender:", ["Men", "Women"], index=0)
    marital_status = st.selectbox("Marital Status:", ["Single", "Married"], index=1)
    employee_status = st.selectbox("Employment Status:", ["employed", "self employed", "unemployed", "student"], index=0)
    residence_type = st.selectbox("Residence Type:", ["MORTGAGE", "OWN", "RENT"], index=1)
    loan_purpose = st.selectbox("Loan Purpose:", ["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"], index=0)

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "cibil_score": [cibil_score],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "loan_percent_income": [loan_percent_income],
        "active_loans": [active_loans],
        "gender": [1 if gender == "Women" else 0],
        "marital_status": [1 if marital_status == "Married" else 0],
        "employee_status_self_employed": [1 if employee_status == "self employed" else 0],
        "employee_status_unemployed": [1 if employee_status == "unemployed" else 0],
        "employee_status_student": [1 if employee_status == "student" else 0],
        "residence_type_OWN": [1 if residence_type == "OWN" else 0],
        "residence_type_RENT": [1 if residence_type == "RENT" else 0],
        "loan_purpose_Personal": [1 if loan_purpose == "Personal" else 0],
        "loan_purpose_Home_Renovation": [1 if loan_purpose == "Home Renovation" else 0],
        "loan_purpose_Education": [1 if loan_purpose == "Education" else 0],
        "loan_purpose_Vehicle": [1 if loan_purpose == "Vehicle" else 0],
    })

    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Loan Prediction
    if st.button("Predict Loan Status"):
        try:
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            if prediction[0] == 1:
                st.markdown("#### Loan Rejected ❌")
                st.error(f"Rejection Probability: {prediction_proba[0][1]:.2f}")
            else:
                st.markdown("#### Loan Approved ✅")
                st.success(f"Approval Probability: {prediction_proba[0][0]:.2f}")

            # Generate PDF report
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', size=12)

            pdf.cell(200, 10, txt="Loan Approval Prediction Report", ln=True, align="C")
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Prediction: {'Approved' if prediction[0] == 0 else 'Rejected'}", ln=True)
            pdf.cell(200, 10, txt=f"Approval Probability: {prediction_proba[0][0]:.2f}", ln=True)
            pdf.cell(200, 10, txt=f"Rejection Probability: {prediction_proba[0][1]:.2f}", ln=True)
            pdf.ln(10)

            details = [
                f"CIBIL Score: {cibil_score}",
                f"Annual Income: INR {income_annum}",
                f"Loan Amount: INR {loan_amount}",
                f"Loan Term: {loan_term} months",
                f"Loan Percent of Income: {loan_percent_income}%",
                f"Number of Active Loans: {active_loans}",
                f"Gender: {gender}",
                f"Marital Status: {marital_status}",
                f"Employment Status: {employee_status}",
                f"Residence Type: {residence_type}",
                f"Loan Purpose: {loan_purpose}",
            ]
            for detail in details:
                pdf.cell(200, 10, txt=detail, ln=True)

            buffer = BytesIO()
            pdf.output(buffer, 'S')
            buffer.seek(0)

            st.download_button(
                label="Download Report as PDF",
                data=buffer,
                file_name="loan_prediction_report.pdf",
                mime="application/pdf"
            )

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Visualization: Credit Score vs Approval Probability
st.markdown("### Credit Score vs Approval Likelihood")
credit_scores = np.arange(300, 901, 50)

# Prepare approval probabilities
approval_probs = []
for score in credit_scores:
    temp_data = pd.DataFrame({
        "cibil_score": [score],
        "income_annum": [500000],  # Example constant income
        "loan_amount": [200000],  # Example constant loan amount
        "loan_term": [24],
        "loan_percent_income": [20],
        "active_loans": [1],
        "gender": [0],
        "marital_status": [1],
        "employee_status_self_employed": [0],
        "employee_status_unemployed": [0],
        "employee_status_student": [0],
        "residence_type_OWN": [1],
        "residence_type_RENT": [0],
        "loan_purpose_Personal": [0],
        "loan_purpose_Home_Renovation": [0],
        "loan_purpose_Education": [0],
        "loan_purpose_Vehicle": [1],
    })

    temp_data = temp_data.reindex(columns=model.feature_names_in_, fill_value=0)
    prob = model.predict_proba(temp_data)[0][0]
    approval_probs.append(prob)

plt.figure(figsize=(8, 4))
plt.plot(credit_scores, approval_probs, marker='o', color="blue", label="Approval Probability")
plt.title("Credit Score vs Approval Likelihood")
plt.xlabel("CIBIL Score")
plt.ylabel("Approval Probability")
plt.grid()
plt.legend()
st.pyplot(plt)

# Expandable FAQ Section
with st.expander("❓ Frequently Asked Questions"):
    st.write("""
    1. **What is a good CIBIL score?**
       A score above 750 is considered excellent for loan approvals.
    2. **What documents are required?**
       ID Proof, Address Proof, Income Proof, and Loan Purpose Declaration.
    """)

# Footer
st.markdown(
    """
    <footer>
        <p>© 2025 AI Loan Application System. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
