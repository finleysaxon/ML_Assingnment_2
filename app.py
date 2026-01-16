import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Airline Passenger Satisfaction",
    layout="wide"
)

st.title("✈️ Airline Passenger Satisfaction Prediction")
st.write("Binary Classification using Multiple ML Models")

# -------------------------------
# Load Models & Scaler
# -------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
        "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
        "KNN": pickle.load(open("models/knn.pkl", "rb")),
        "Naive Bayes": pickle.load(open("models/naive_bayes.pkl", "rb")),
        "Random Forest": pickle.load(open("models/random_forest.pkl", "rb")),
        "XGBoost": pickle.load(open("models/xgboost.pkl", "rb"))
    }

    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))

    return models, scaler, feature_columns


models, scaler, feature_columns = load_models()


# -------------------------------
# Upload Dataset
# -------------------------------
st.sidebar.header("Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (test data only)",
    type=["csv"]
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

# -------------------------------
# Main Logic
# -------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Drop unused columns
    df.drop(columns=['id', 'Unnamed: 0'], inplace=True, errors='ignore')

    # Fill Arrival Delay missing values
    if 'Arrival Delay in Minutes' in df.columns:
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(
        df['Arrival Delay in Minutes'].median()
    )

    # Fill any remaining numeric NaNs
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Encode target
    y_true = df['satisfaction'].map({
        'neutral or dissatisfied': 0,
        'satisfied': 1
    })

    X = df.drop(columns=['satisfaction'])

    # One-hot encoding
    X = pd.get_dummies(
        X,
        columns=['Gender', 'Customer Type', 'Type of Travel', 'Class'],
        drop_first=True
    )

    # Align columns
    model = models[model_choice]
    X = X.reindex(columns=feature_columns, fill_value=0)


   # Models that REQUIRE scaling
    scale_models = ["Logistic Regression", "KNN", "Naive Bayes"]

    if model_choice in scale_models:
        X_input = scaler.transform(X)
    else:
        X_input = X.values  # raw, unscaled data

    #Predict
    y_pred = model.predict(X_input)
    y_prob = model.predict_proba(X_input)[:, 1]


    # Metrics
    st.subheader("📊 Model Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
    col2.metric("Precision", round(precision_score(y_true, y_pred), 4))
    col3.metric("Recall", round(recall_score(y_true, y_pred), 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y_true, y_pred), 4))
    col5.metric("AUC", round(roc_auc_score(y_true, y_prob), 4))
    col6.metric("MCC", round(matthews_corrcoef(y_true, y_pred), 4))

    # Confusion Matrix
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
