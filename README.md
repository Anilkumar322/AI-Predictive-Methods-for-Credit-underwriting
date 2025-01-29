# AI Predictive Methods for Credit Underwriting

This project is a **Streamlit-based AI Predictive Methods for Credit Underwriting application** that utilizes a machine learning model to predict whether a loan application will be approved or rejected based on user-provided inputs. The app generates a downloadable PDF report with detailed results and insights.

---

## Features

- **Interactive Web App**: Built with Streamlit for an intuitive and responsive user interface.
- **User Inputs**: Input fields for CIBIL score, income, loan amount, loan term, and other details.
- **Prediction**: Uses a pre-trained machine learning model to predict loan approval status.
- **Downloadable Report**: Generates a professional PDF report summarizing predictions and input details.
- **Styling**: Custom CSS for enhanced user experience.

---

## Technologies Used

- **Python**: Programming language for building the application.
- **Streamlit**: Framework for creating the interactive web app.
- **FPDF**: Library for generating downloadable PDF reports.
- **pandas**: For data preparation and handling user inputs.
- **matplotlib**: For visualizations (optional).
- **joblib**: For loading the pre-trained machine learning model.

---

## Prerequisites

Ensure the following are installed on your system:

- Python 3.7+
- pip (Python package installer)

Install the required libraries by running:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
streamlit
pandas
matplotlib
joblib
fpdf
```

---

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Anilkumar322/AI-Predictive-Methods-for-Credit-underwriting.git
   cd AI-Predictive-Methods-for-Credit-underwriting
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. Open the provided local URL in your browser to access the app.

---

## File Structure

- `streamlit_app.py`: Main application file.
- `requirements.txt`: List of required Python libraries.
- `best_features_model.pkl`: Pre-trained machine learning model file.

---

## User Inputs

The app provides the following input fields in the sidebar:

1. **CIBIL Score** (300-900)
2. **Annual Income** (INR)
3. **Loan Amount** (INR)
4. **Loan Term** (in months)
5. **Loan Percent of Income** (%)
6. **Number of Active Loans**
7. **Gender**
8. **Marital Status**
9. **Employment Status**
10. **Residence Type**
11. **Loan Purpose**

---

## Output

- **Streamlit Link**: [Click here to access the app](https://ai-predictive-methods-for-credit-underwriting-e6mhvscqrvuabfjy.streamlit.app/)
- **Loan Status**: Displays whether the loan is approved or rejected.
- **Prediction Probabilities**: Shows the probability of approval and rejection.
- **PDF Report**: Generates a detailed downloadable PDF report containing:
  - Prediction results.
  - Input details provided by the user.
  - Summary of probabilities.

---

## Deployment

To deploy the app on a platform like **Streamlit Cloud**:

1. Push the project to GitHub.
2. Connect your GitHub repository to Streamlit Cloud.
3. Ensure the `requirements.txt` file is present for dependency installation.
4. Deploy and access your app via the Streamlit Cloud link.

---

## Troubleshooting

### Missing Library Error
If you encounter an error like `ModuleNotFoundError: No module named 'fpdf'`, run:
```bash
pip install fpdf
```

### Unicode Character Error (₹ Symbol)
Ensure you have a Unicode-compatible font (e.g., `FreeSerif.ttf`) in your working directory. Update the PDF font registration in the code if necessary.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributors

- **Bollapalli Anil Kumar** (Developer)
- [GitHub Profile](https://github.com/Anilkumar322)

---

## Feedback

For any issues or suggestions, please open an issue on the [GitHub repository](https://github.com/Anilkumar322/AI-Predictive-Methods-for-Credit-underwriting.git).

