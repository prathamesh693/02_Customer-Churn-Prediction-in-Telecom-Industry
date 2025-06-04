import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from PIL import Image

# Set page config
st.set_page_config(page_title="Telecom Customer Churn Analysis", layout="wide")

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv("02_Data/preprocessed_data.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("05_Model/best_model_xgboost.pkl")
    label_encoders = joblib.load("05_Model/label_encoders.pkl")
    return model, label_encoders

# Load assets
df = load_data()
model, label_encoders = load_model()

# Header
# Custom CSS with animations and styling
st.markdown("""
<style>
    .header-container {
        position: relative;
        width: 100%;
        margin-bottom: 2rem;
    }
    .header-title {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        text-align: center;
        width: 100%;
        padding: 0 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .header-subtitle {
        font-size: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with enhanced visuals
header_img = Image.open("Image/header_telecom.jpg")

# Create a container for the header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(header_img, use_column_width=True)

# Helper function to convert image to base64
def image_to_base64(img):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()
st.title("Telecom Customer Churn Analysis Dashboard")

# Sidebar
st.sidebar.image("Image/telecom_icon.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Overview", "Customer Analysis", "Churn Prediction"])
# Sidebar with custom CSS
st.sidebar.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #f0f2f5;
        padding: 20px;
        border-radius: 10px;
    }
    .sidebar .sidebar-content h1 {
        font-size: 1.5rem;
        color: #333;
    }
    .sidebar .sidebar-content a {
        color: #3498db;
        text-decoration: none;
    }
    .sidebar .sidebar-content a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Overview Page
if page == "Overview":
    st.header("Project Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        # Key Metrics
        st.subheader("Key Metrics")
        total_customers = len(df)
        churn_rate = (df['Churn'].mean() * 100)
        avg_tenure = df['tenure'].mean()
        avg_monthly_charges = df['MonthlyCharges'].mean()
        
        st.metric("Total Customers", f"{total_customers:,}")
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
        st.metric("Average Tenure", f"{avg_tenure:.1f} months")
        st.metric("Average Monthly Charges", f"${avg_monthly_charges:.2f}")
    
    with col2:
        # Churn Distribution
        fig = px.pie(df, names='Churn', title='Customer Churn Distribution',
                     color_discrete_sequence=['#3498db', '#e74c3c'])
        st.plotly_chart(fig, use_container_width=True)

# Customer Analysis Page
if page == "Customer Analysis":
    st.header("Customer Insights")
    
    # Tenure vs Monthly Charges by Churn
    fig = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                     title='Tenure vs Monthly Charges by Churn Status',
                     labels={'tenure': 'Tenure (months)', 'MonthlyCharges': 'Monthly Charges ($)'})
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Contract Type Distribution
        contract_dist = px.bar(df, x='Contract', title='Distribution by Contract Type',
                              color='Churn', barmode='group')
        st.plotly_chart(contract_dist, use_container_width=True)
    
    with col2:
        # Payment Method Analysis
        payment_dist = px.bar(df, x='PaymentMethod', title='Distribution by Payment Method',
                             color='Churn', barmode='group')
        st.plotly_chart(payment_dist, use_container_width=True)
    
    # Correlation Heatmap
    corr = df.corr()
    fig = px.imshow(corr, title='Feature Correlation Heatmap',
                    color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)

# Churn Prediction Page
if page == "Churn Prediction":
    st.header("Customer Churn Prediction")
    st.write("Enter customer information to predict churn probability")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior_citizen = st.selectbox("Senior Citizen", ['No', 'Yes'])
        partner = st.selectbox("Partner", ['Yes', 'No'])
        dependents = st.selectbox("Dependents", ['Yes', 'No'])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
        multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])
    
    with col2:
        internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        device_protection = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
        tech_support = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
    
    with col3:
        streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
        contract = st.selectbox("Contract Type", ['Month-to-month', '1 year', '2 year'])
        paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
        payment_method = st.selectbox("Payment Method", 
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly_charges = st.number_input("Monthly Charges ($)", 0, 1000, 50)
        total_charges = st.number_input("Total Charges ($)", 0, 10000, monthly_charges * tenure)
    
    if st.button("Predict Churn Probability"):
        # Prepare input data with all required features
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Encode categorical variables
        for col in input_data.columns:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])
            elif col == 'SeniorCitizen':
                input_data[col] = 1 if input_data[col].iloc[0] == 'Yes' else 0
        
        # Make prediction
        try:
            churn_prob = model.predict_proba(input_data)[0][1]
            
            # Display result with gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = churn_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            if churn_prob < 0.3:
                st.success("Low churn risk! 🎉")
            elif churn_prob < 0.7:
                st.warning("Moderate churn risk! ⚠️")
            else:
                st.error("High churn risk! 🚨")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please check if all input values are valid.")

# Footer
st.markdown("---")
st.markdown("### About")
st.write("""
This dashboard provides insights into customer churn patterns and allows for real-time churn prediction 
 using machine learning. The model is trained on historical customer data and uses various features to 
 predict the likelihood of customer churn.
""")