import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
file_path = 'C:\\Credit_Underwriting\\credit_underwriting1.csv'  # Update with your local file path
print(f"Loading dataset from: {file_path}")
dataset = pd.read_csv(file_path)

# Step 2: Clean the dataset (adjust column names and preprocessing as needed)
dataset.columns = dataset.columns.str.strip()
dataset['loan_interest'] = dataset['loan_interest'].replace({',': '.'}, regex=True).astype(float)
dataset['loan_percent_income'] = dataset['loan_percent_income'].replace({',': '.'}, regex=True).astype(float)
categorical_columns = ['gender', 'marital_status', 'employee_status', 'residence_type', 'loan_purpose']
X = pd.get_dummies(dataset.drop(columns=['loan_id', 'loan_status']), columns=categorical_columns, drop_first=True)
y = dataset['loan_status']

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Step 5: Create the ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('Random Forest', rf_model),
        ('Gradient Boosting', gb_model)
    ],
    voting='soft'
)
ensemble_model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = ensemble_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the model
model_path = 'ensemble_model.pkl'  # Save the model to the current directory
joblib.dump(ensemble_model, model_path)
print(f"Model saved to: {model_path}")
