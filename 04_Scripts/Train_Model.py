import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import joblib
import os

# Load data
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/02_Data/Preprocessed_data.csv")
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize models
dt_model = DecisionTreeClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
svm_model = SVC(probability=True, random_state=42)

models = {
    'Decision Tree': dt_model,
    'XGBoost': xgb_model,
    'SVM': svm_model
}

best_model_name = None
best_model = None
best_auc = 0

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"{name} ROC AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

print(f"\nBest model: {best_model_name} with ROC AUC = {best_auc:.4f}")

# ------- Save the best model, Train and Test dataset -------
save_dir = "R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry"
model_save_path = os.path.join(save_dir, "05_Model", f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl")
joblib.dump(best_model, model_save_path)
print(f"Saved the best model as {model_save_path}")

# Save train data
train_df = X_train.copy()
train_df['Actual'] = y_train.values
train_data_path = os.path.join(save_dir, "06_Outputs", "train_data.csv")
train_df.to_csv(train_data_path, index=False)

# Save test data
test_df = X_test.copy()
test_df['Actual'] = y_test.values
test_data_path = os.path.join(save_dir, "06_Outputs", "test_data.csv")
test_df.to_csv(test_data_path, index=False)

# -------- Feature Selection --------
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]

# Create a DataFrame with selected features and target
df_selected = X[selected_features].copy()
df_selected['Churn'] = y.values

# -------- PCA --------
pca = PCA(n_components=5)
X_pca = pca.fit_transform(df_selected.drop('Churn', axis=1))

# Save the selector and PCA objects
selector_path = os.path.join(save_dir, "05_Model", "selector.pkl")
pca_path = os.path.join(save_dir, "05_Model", "pca.pkl")
joblib.dump(selector, selector_path)
joblib.dump(pca, pca_path)

print(f"Saved feature selector as {selector_path}")
print(f"Saved PCA object as {pca_path}")