import os, sys

# must be set BEFORE importing anything else
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource(show_spinner="Loading models...")
def get_models():
    from predict import load_models
    return load_models()

models = get_models()

st.title("💳 Credit Risk Predictor")
st.caption("Home Credit Default Risk — Stacked Ensemble (LGB + XGB + CatBoost) | OOF AUC: 0.7881")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    gender         = st.selectbox("Gender", ["M", "F"])
    age            = st.slider("Age", 18, 70, 35)
    education      = st.selectbox("Education", [
        "Higher education", "Secondary / secondary special",
        "Incomplete higher", "Lower secondary", "Academic degree"
    ])
    family_status  = st.selectbox("Family Status", [
        "Married", "Single / not married", "Civil marriage", "Separated", "Widow"
    ])
    family_members = st.slider("Family Members", 1, 10, 2)

with col2:
    st.subheader("Employment & Assets")
    employment_years = st.slider("Years Employed", 0, 40, 5)
    income           = st.number_input("Annual Income", min_value=1, value=300000, step=10000)
    owns_car         = st.selectbox("Owns Car", ["Y", "N"])
    owns_house       = st.selectbox("Owns House", ["Y", "N"])

with col3:
    st.subheader("Loan Details")
    loan_amount  = st.number_input("Loan Amount", min_value=1, value=500000, step=10000)
    annuity      = st.number_input("Monthly Annuity", min_value=1, value=25000, step=1000)
    ext_source_1 = st.slider("External Score 1", 0.0, 1.0, 0.5, step=0.01)
    ext_source_2 = st.slider("External Score 2", 0.0, 1.0, 0.5, step=0.01)
    ext_source_3 = st.slider("External Score 3", 0.0, 1.0, 0.5, step=0.01)

st.divider()

if st.button("🔍 Predict Default Risk", use_container_width=True, type="primary"):

    from predict import predict_single, BEST_THRESHOLD

    app_df = pd.DataFrame({
        "SK_ID_CURR":          [999999],
        "CODE_GENDER":         [gender],
        "DAYS_BIRTH":          [-age * 365],
        "DAYS_EMPLOYED":       [-employment_years * 365 if employment_years > 0 else 365243],
        "AMT_INCOME_TOTAL":    [float(income)],
        "AMT_CREDIT":          [float(loan_amount)],
        "AMT_ANNUITY":         [float(annuity)],
        "CNT_FAM_MEMBERS":     [float(family_members)],
        "FLAG_OWN_CAR":        [owns_car],
        "FLAG_OWN_REALTY":     [owns_house],
        "NAME_EDUCATION_TYPE": [education],
        "NAME_FAMILY_STATUS":  [family_status],
        "OWN_CAR_AGE":         [None],
        "EXT_SOURCE_1":        [ext_source_1],
        "EXT_SOURCE_2":        [ext_source_2],
        "EXT_SOURCE_3":        [ext_source_3],
    })

    with st.spinner("Running prediction..."):
        result = predict_single(app_df, models=models)

    risk       = result["stacked"]
    label      = result["risk_label"]
    is_default = result["is_default"]

    emoji = "✅" if label == "Low Risk" else "⚠️" if label == "Medium Risk" else "🚨"

    st.subheader("Results")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Stacked Score", f"{risk:.4f}")
    with r2:
        st.metric("Risk Label", f"{emoji} {label}")
    with r3:
        st.metric("Default Flag", "YES" if is_default else "NO")
    with r4:
        st.metric("Decision", "❌ Reject" if is_default else "✅ Approve")

    st.progress(min(risk, 1.0), text=f"Default probability: {risk:.1%}")

    st.divider()
    st.subheader("Model Breakdown")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("LightGBM", f"{result['lgb']:.4f}")
    with m2:
        st.metric("XGBoost", f"{result['xgb']:.4f}")
    with m3:
        st.metric("CatBoost", f"{result['cat']:.4f}")

    with st.expander("📊 Engineered Features"):
        st.write(f"**Credit / Income Ratio:** {round(loan_amount / income, 4)}")
        st.write(f"**Annuity / Income Ratio:** {round(annuity / income, 4)}")
        st.write(f"**EXT Source Mean:** {round((ext_source_1 + ext_source_2 + ext_source_3) / 3, 4)}")
        st.write(f"**Income per Person:** {round(income / family_members, 2)}")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("Predicts probability of loan default using a stacked ensemble trained on the Home Credit Default Risk dataset.")
    st.divider()
    st.write("**Model Performance (OOF)**")
    st.write("LightGBM:  0.7821")
    st.write("XGBoost:   0.7851")
    st.write("CatBoost:  0.7864")
    st.write("**Stacked:  0.7881**")
    st.divider()
    st.write("**Threshold:** 0.115")
    st.write("**Recall:** ~44% of defaults caught")