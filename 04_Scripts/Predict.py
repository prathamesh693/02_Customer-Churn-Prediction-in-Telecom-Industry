import pandas as pd
import joblib
import os

# Load test data
df=pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/06_Outputs/test_data.csv")

# Drop Actual column if present
if "Actual" in df.columns:
    X = df.drop("Actual", axis=1)
    print("Removed 'Actual' column for prediction.")
else:
    X = df

# Load trained model
model = joblib.load("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/05_Model/best_model_xgboost.pkl")
# Predict
preds = model.predict(X)
pred_proba = model.predict_proba(X)[:, 1]

# Append predictions to DataFrame
df["Churn_Prediction"] = preds
df["Churn_Probability"] = pred_proba

# Save prediction results 
df.to_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/06_Outputs/predict_output.csv", index=False)
print("Predictions saved")