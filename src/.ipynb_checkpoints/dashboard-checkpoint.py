import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from train import load_data
from features import (
    preprocess_application,
    bureau_features,
    bureau_balance_features,
    previous_application_features,
    pos_features,
    installment_features,
    credit_card_features,
)

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("🏦 Credit Risk Assessment Dashboard")

st.write("Upload applicant data and evaluate risk using the trained model.")

# Load model
@st.cache_resource
def load_model():
    pipeline = joblib.load("models/lgb_model.pkl")
    return pipeline

model_pipeline = load_model()

preprocess = model_pipeline.named_steps["preprocess"]
model = model_pipeline.named_steps["model"]

# File uploader
uploaded_file = st.file_uploader("Upload applicant CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    st.write("Processing features...")

    # Minimal preprocessing (assuming same format as training)
    df = preprocess_application(df)

    st.write("Transforming features...")

    X_transformed = preprocess.transform(df)

    feature_names = preprocess.get_feature_names_out()

    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    st.write("Generating predictions...")

    preds = model.predict_proba(X_transformed)[:, 1]

    df["Risk Score"] = preds

    st.subheader("Risk Predictions")

    st.dataframe(df[["SK_ID_CURR", "Risk Score"]])

    st.write("Explaining prediction using SHAP...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_transformed)

    # Waterfall plot for first applicant
    st.subheader("Risk Explanation")

    fig = plt.figure(figsize=(10,5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)

    st.pyplot(fig)

    # Global importance
    st.subheader("Feature Importance")

    fig2 = plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_transformed, max_display=15, show=False)

    st.pyplot(fig2)

else:

    st.info("Upload a CSV file containing applicant data to generate predictions.")