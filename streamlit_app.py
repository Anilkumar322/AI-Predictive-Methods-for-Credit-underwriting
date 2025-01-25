import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

# Set page configuration
st.set_page_config(
    page_title="AI Predictive Methods for Credit underwriting",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
        body {
            background-color: #f4f7fa;
            font-family: 'Arial', sans-serif;
        }
        .header-container {
            background: linear-gradient(to right, #4CAF50, #45a049);
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px gray;
            text-align: center;
            margin-bottom: 20px;
        }
        .header-container h1 {
            font-size: 36px;
        }
        .header-container p {
            font-size: 18px;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px lightgray;
            margin-bottom: 20px;
        }
        .result-container {
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            font-weight: bold;
        }
        .result-approved {
            background-color: #e7f9e7;
            color: green;
            border: 2px solid green;
        }
        .result-rejected {
            background-color: #fde9e9;
            color: red;
            border: 2px solid red;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        footer {
            text-align: center;
            margin-top: 50px;
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
        <h1>AI Predictive Methods for Credit underwriting</h1>
        <p>Revolutionizing credit underwriting with AI-driven predictive analytics for smarter, faster decisions!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model_path = 'best_features_model.pkl'  # Path to the trained model
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Sidebar for input fields
with st.sidebar:
    st.markdown(
        """
        <div class="card">
            <h3>Input Loan Details</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    cibil_score = st.slider("CIBIL Score (300-900):", min_value=300, max_value=900, step=1, value=750)
    income_annum = st.number_input("Annual Income (INR):", min_value=0, step=10000, value=5000000)
    loan_amount = st.number_input("Loan Amount (INR):", min_value=0, step=10000, value=2000000)
    loan_term = st.number_input("Loan Term (Months):", min_value=0, step=1, value=24)
    loan_percent_income = st.number_input("Loan Percent of Income (%):", min_value=0.0, step=0.1, value=20.0)
    active_loans = st.number_input("Number of Active Loans:", min_value=0, step=1, value=1)

    gender = st.selectbox("Gender:", ["Men", "Women"], index=0)
    marital_status = st.selectbox("Marital Status:", ["Single", "Married"], index=1)
    employee_status = st.selectbox("Employment Status:", ["employed", "self employed", "unemployed", "student"], index=0)
    residence_type = st.selectbox("Residence Type:", ["MORTGAGE", "OWN", "RENT"], index=1)
    loan_purpose = st.selectbox("Loan Purpose:", ["Vehicle", "Personal", "Home Renovation", "Education", "Medical", "Other"], index=0)

# Prepare the input data
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

# Align input data with model features
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction button
if st.button("Predict Loan Status"):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            status = "Rejected"
            st.markdown(
                """
                <div class="result-container result-rejected">
                    <h1>Loan Rejected ❌</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.error(f"Rejection Probability: {prediction_proba[0][1]:.2f}")
        else:
            status = "Approved"
            st.markdown(
                """
                <div class="result-container result-approved">
                    <h1>Loan Approved ✅</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.success(f"Approval Probability: {prediction_proba[0][0]:.2f}")
# Generate PDF report
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', size=12)

# Add content to the PDF
pdf.cell(200, 10, txt="Loan Approval Prediction Report", ln=True, align="C")
pdf.ln(10)
pdf.cell(200, 10, txt=f"Prediction: {'Approved' if prediction[0] == 0 else 'Rejected'}", ln=True)
pdf.cell(200, 10, txt=f"Approval Probability: {prediction_proba[0][0]:.2f}", ln=True)
pdf.cell(200, 10, txt=f"Rejection Probability: {prediction_proba[0][1]:.2f}", ln=True)

pdf.ln(10)
pdf.cell(200, 10, txt="Details:", ln=True)
pdf.ln(5)

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

# Write the PDF to a BytesIO object
buffer = BytesIO()
pdf.output(buffer)  # This writes the PDF data to the buffer
buffer.seek(0)  # Reset the buffer's pointer to the beginning

# Allow the user to download the PDF
st.download_button(
    label="Download Report as PDF",
    data=buffer,
    file_name="loan_prediction_report.pdf",
    mime="application/pdf"
)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown(
    """
    <footer>
        <p>© 2025 AI Predictive Methods for Credit underwriting. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
