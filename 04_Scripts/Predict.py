import pandas as pd
import joblib

# ----------- Step 1: Load the test data -----------
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/06_Outputs/test_data.csv")
print(f"Number of samples to predict: {len(df)}")

# Separate actual labels if present
if 'Actual' in df.columns:
    X = df.drop('Actual', axis=1)
    print("Dropped 'Actual' column from test data for prediction.")
else:
    X = df
    print("'Actual' column not found in test data, using all columns as features.")

# ----------- Step 2: Load the saved model -----------
model = joblib.load("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/05_Model/best_model_xgboost.pkl")
print("Loaded model from 05_Model folder")

# ----------- Step 3: Make predictions -----------
preds = model.predict(X)
pred_proba = model.predict_proba(X)[:, 1]

print(f"Made predictions on {len(preds)} samples.")

# ----------- Step 4: Add predictions back to DataFrame -----------
df['Churn_Prediction'] = preds
df['Churn_Probability'] = pred_proba

# ----------- Step 5: Save results -----------
df.to_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/06_Outputs/predict_output.csv", index=False)
print("Predictions saved to 06_Outputs Folder!")
