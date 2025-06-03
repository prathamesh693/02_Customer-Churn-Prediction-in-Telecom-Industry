import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/02_Data/Telco-Customer-Churn.csv")

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle missing values
if df['TotalCharges'].isnull().sum() > 0:
    print(f"Filling {df['TotalCharges'].isnull().sum()} missing TotalCharges with median")
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save processed data
df.to_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/02_Data/preprocessed_data.csv", index=False)
print("Saved preprocessed data to 02_Data/preprocessed_data.csv")

# Save label encoders
joblib.dump(label_encoders, "R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/05_Model/label_encoders.pkl")
print("Saved label encoders.")