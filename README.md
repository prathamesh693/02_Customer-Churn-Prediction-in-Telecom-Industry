# Telecom Customer Churn Prediction

## Overview
Predict customer churn in telecom using Decision Tree, XGBoost, and SVM.

## File Description
- `Preprocessing.py`: Preprocess raw data and save `preprossed-data.csv`
- `Train_model.py`: Train models and save best model as `one-model-saved.pkl`
- `predict.py`: Predict churn on test data and save results
- `app.py`: Streamlit app for interactive predictions
- `eda.ipynb`, `modeling.ipynb`, `evaluation.ipynb`: Exploratory, modeling and evaluation notebooks

## Usage
1. Prepare your `train-data.csv` and `test-data.csv`.
2. Run preprocessing:  
   `python Preprocessing.py`
3. Train model:  
   `python Train_model.py`
4. Predict churn on test data:  
   `python predict.py`
5. Run app:  
   `streamlit run app.py`
## Dataset
Download from [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)