import streamlit as st
import pandas as pd
import joblib

st.title("Telecom Customer Churn Prediction App")

uploaded_file = st.file_uploader("Upload CSV file with customer data", type=['csv'])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())
    
    # Load the saved model (change the path as needed)
    model_path = "R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/05_Model/best_model_xgboost.pkl"
    model = joblib.load(model_path)
    
    # Assuming data is already preprocessed exactly like training data
    # If target column is present, drop it
    if 'Churn' in data.columns:
        X = data.drop('Churn', axis=1)
    else:
        X = data
    
    # Make predictions
    preds = model.predict(X)
    pred_proba = model.predict_proba(X)[:, 1]
    
    # Add prediction results to dataframe
    data['Churn_Prediction'] = preds
    data['Churn_Probability'] = pred_proba
    
    st.write("Prediction Results:")
    st.dataframe(data[['Churn_Prediction', 'Churn_Probability']])