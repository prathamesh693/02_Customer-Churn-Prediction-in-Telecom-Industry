# 📉 Customer Churn Prediction in Telecom Industry  
**🎯 Machine Learning to Enhance Customer Retention**

This project aims to predict customer churn in the telecom industry using machine learning models such as Decision Tree, XGBoost, and SVM. By identifying at-risk customers early, telecom companies can take proactive steps to retain them and reduce revenue loss. The end-to-end pipeline includes preprocessing, model building, evaluation, and deployment through a Streamlit web app.

---

## 📚 Table of Contents  
- [Problem Statement](#problem-statement)  
- [Objective](#objective)  
- [Challenges](#challenges)  
- [Project Lifecycle](#project-lifecycle)  
- [File Description](#file-description)  
- [Usage](#usage)  
- [Tools and Technologies](#tools-and-technologies)  
- [Success Criteria](#success-criteria)  
- [Expected Outcome](#expected-outcome)  
- [References](#references)  
- [Connect With Me](#connect-with-me)

---

## 📌 Problem Statement  
Customer churn significantly impacts telecom business growth. Retaining existing customers is more cost-effective than acquiring new ones. This project aims to develop a churn prediction model using customer usage, demographics, and service data to enable personalized retention strategies.

---

## 🎯 Objective  
- Predict whether a customer is likely to churn using ML algorithms  
- Provide actionable insights into churn drivers  
- Deploy a user-friendly prediction tool using Streamlit  

---

## ⚠️ Challenges  
- Imbalanced dataset: Fewer churners vs. non-churners  
- Missing/inconsistent data in customer records  
- Selecting the most impactful features  
- Ensuring robust performance on unseen data  
- Providing explainability for business decision-making  

---

## 🛠️ Project Lifecycle  
1. **Data Collection**  
   - Kaggle’s Telco Customer Churn dataset  
2. **Data Preprocessing**  
   - Handling nulls, encoding categoricals, scaling features  
3. **Exploratory Data Analysis (EDA)**  
   - Visualizing churn rates by tenure, services, and demographics  
4. **Model Building**  
   - Train Decision Tree, XGBoost, and SVM classifiers  
5. **Model Evaluation**  
   - Accuracy, ROC-AUC, precision, recall, F1-score  
6. **Model Deployment**  
   - Streamlit app for churn prediction  
7. **Monitoring & Maintenance**  
   - Model performance tracking and periodic updates  

---

## 📁 File Description  
- `Preprocessing.py` – Prepares and cleans raw input data  
- `Train_model.py` – Trains models and saves best one  
- `predict.py` – Runs model inference on new/test data  
- `app.py` – Interactive Streamlit web app  
- `eda.ipynb` – Exploratory data analysis notebook  
- `modeling.ipynb` – Model training and comparison notebook  
- `evaluation.ipynb` – Evaluation metrics and visualizations  

---

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

---

## 💻 Tools and Technologies  

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-EC0000?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-44A8B3?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Joblib-008000?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/VSCode-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white" />
  <img src="https://img.shields.io/badge/Spyder-FF0000?style=for-the-badge&logo=python&logoColor=white" />
</p>

---

## Success Criteria
- Achieve high classification accuracy and ROC-AUC score on test data.  
- Demonstrate the model’s ability to generalize well to unseen data.  
- Provide actionable insights into churn drivers via EDA and feature importance.  
- Deliver an easy-to-use Streamlit app for business users to predict churn on new data.  
- Maintain reproducibility and code modularity for future enhancements.

---

## 🔗 References (📊Dataset)
Download from [Kaggle Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 🤝 Connect With Me  
**[![LinkedIn](https://img.shields.io/badge/LinkedIn-Prathamesh%20Jadhav-blue?logo=linkedin)](https://www.linkedin.com/in/prathamesh-jadhav-78b02523a/) [![GitHub](https://img.shields.io/badge/GitHub-Prathamesh%20Jadhav-2b3137?logo=github)](https://github.com/prathamesh693)**