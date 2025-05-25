# Customer Churn Prediction in Telecom Industry

## 📗Project Overview
Predict customer churn in telecom using Decision Tree, XGBoost, and SVM.

## File Description
- `Preprocessing.py`: Preprocess raw data and save `preprossed-data.csv`
- `Train_model.py`: Train models and save best model as `one-model-saved.pkl`
- `predict.py`: Predict churn on test data and save results
- `app.py`: Streamlit app for interactive predictions
- `eda.ipynb`, `modeling.ipynb`, `evaluation.ipynb`: Exploratory, modeling and evaluation notebooks

## ⚙️Usage
1. Prepare your `train-data.csv` and `test-data.csv`.
2. Run preprocessing:  
   `python Preprocessing.py`
3. Train model:  
   `python Train_model.py`
4. Predict churn on test data:  
   `python predict.py`
5. Run app:  
   `streamlit run app.py`

## 🛠️Tools & Technologies
1. 🧪 Data Processing & Analysis
   -- **Pandas** – Data manipulation and analysis
   -- **NumPy** – Numerical computing
2. 🔄 Preprocessing & Feature Engineering
   --**Scikit-learn**
   **LabelEncoder, MinMaxScaler – Encoding and scaling SelectKBest, PCA – Dimensionality reduction, train_test_split – Data splitting** 
3. 🤖 Machine Learning Models
   --**Scikit-learn** 
   1. DecisionTreeClassifier
   2. SVC (Support Vector Classifier)
   3. XGBoost
   4. XGBClassifier – Gradient Boosting Model
4. 📈 Data Visualization
   --**Matplotlib** – Basic plotting
   --**Seaborn** – Statistical and heatmap visualizations
   --**Power BI** - Creating a Dashboard
## 📊Dataset
Download from [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)