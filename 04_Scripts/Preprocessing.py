import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import joblib

# Load dataset
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/02_Data/Telco-Customer-Churn.csv")

# Drop customerID
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Replace 'No internet service' and 'No phone service' with 'No'
cols_to_clean = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
for col in cols_to_clean:
    df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

# Label encode binary columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'] + cols_to_clean
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-hot encode multi-class columns
multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod']
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

# Remove outliers using IQR
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Normalize numeric columns
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Feature selection
X = df.drop('Churn', axis=1)
y = df['Churn']
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
df_selected = pd.DataFrame(X_new, columns=selected_features)
df_selected['Churn'] = y.values

# PCA dimensionality reduction
pca = PCA(n_components=5)
X_pca = pca.fit_transform(df_selected.drop('Churn', axis=1))
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(5)])
df_pca['Churn'] = df_selected['Churn'].values

# Save final dataset
df_pca.to_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/02_Data/Preprocessed_data.csv", index=False)
print("Final preprocessed data saved to: Preprocessed_data.csv")

# Save the scaler
joblib.dump(scaler, "R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/05_Model/scaler.pkl")
