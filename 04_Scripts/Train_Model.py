import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import joblib
import os

# Load preprocessed data
df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry/02_Data/preprocessed_data.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# Train and evaluate models
best_model = None
best_model_name = None
best_auc = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"{name} ROC AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# ------- Save the best model -------
save_dir = "R:/Projects/1_Data_Science & ML_Projects/02_Customer Churn Prediction in Telecom Industry"
model_save_path = os.path.join(save_dir, "05_Model", f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl")
joblib.dump(best_model, model_save_path)
print(f"Saved the best model as {model_save_path}")

# Save train and test datasets
X_train_copy = X_train.copy()
X_train_copy["Actual"] = y_train
X_train_copy.to_csv(os.path.join(save_dir, "06_Outputs", "train_data.csv"), index=False)

X_test_copy = X_test.copy()
X_test_copy["Actual"] = y_test
X_test_copy.to_csv(os.path.join(save_dir, "06_Outputs", "test_data.csv"), index=False)

print("Train and test data saved in '06_Outputs/' directory.")