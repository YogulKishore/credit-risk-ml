import os
import gc
import pandas as pd
import numpy as np
import joblib

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (
    preprocess_application,
    bureau_features,
    bureau_balance_features,
    previous_application_features,
    pos_features,
    installment_features,
    credit_card_features
)

# Best threshold from evaluate.py — use this for binary classification
BEST_THRESHOLD = 0.115


# ------------------------------------------------
# Load models once — avoids reloading from disk
# on every single prediction call
# ------------------------------------------------

def load_models():
    lgb_model    = joblib.load("models/lgb_model.pkl")
    xgb_model    = joblib.load("models/xgb_model.pkl")
    cat_model    = joblib.load("models/cat_model.pkl")
    stack_model  = joblib.load("models/stack_model.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return lgb_model, xgb_model, cat_model, stack_model, feature_cols


# ------------------------------------------------
# Batch prediction (Kaggle submission)
# ------------------------------------------------

def predict():

    print("Loading models...")
    lgb_model, xgb_model, cat_model, stack_model, feature_cols = load_models()

    print("Loading data...")
    app_test       = pd.read_csv("data/application_test.csv")
    bureau         = pd.read_csv("data/bureau.csv")
    bureau_balance = pd.read_csv("data/bureau_balance.csv")
    prev_app       = pd.read_csv("data/previous_application.csv")
    pos            = pd.read_csv("data/POS_CASH_balance.csv")
    installments   = pd.read_csv("data/installments_payments.csv")
    credit_card    = pd.read_csv("data/credit_card_balance.csv")

    print("Building features...")
    app_test = preprocess_application(app_test)

    bureau_feat = bureau_features(bureau)
    app_test = app_test.merge(bureau_feat, on="SK_ID_CURR", how="left")
    del bureau, bureau_feat; gc.collect()

    bureau_bal_feat = bureau_balance_features(
        pd.read_csv("data/bureau.csv"), bureau_balance
    )
    app_test = app_test.merge(bureau_bal_feat, on="SK_ID_CURR", how="left")
    del bureau_balance, bureau_bal_feat; gc.collect()

    prev_feat = previous_application_features(prev_app)
    app_test = app_test.merge(prev_feat, on="SK_ID_CURR", how="left")
    del prev_app, prev_feat; gc.collect()

    pos_feat = pos_features(pos)
    app_test = app_test.merge(pos_feat, on="SK_ID_CURR", how="left")
    del pos, pos_feat; gc.collect()

    inst_feat = installment_features(installments)
    app_test = app_test.merge(inst_feat, on="SK_ID_CURR", how="left")
    del installments, inst_feat; gc.collect()

    cc_feat = credit_card_features(credit_card)
    app_test = app_test.merge(cc_feat, on="SK_ID_CURR", how="left")
    del credit_card, cc_feat; gc.collect()

    ids = app_test["SK_ID_CURR"]
    X   = app_test.drop(columns=["SK_ID_CURR"])
    X   = X[feature_cols]

    # fill cat cols for lgb/xgb pipeline
    cat_cols = X.select_dtypes(include=["object", "str"]).columns
    X[cat_cols] = X[cat_cols].fillna("Missing").astype(str)

    print("Generating predictions...")
    lgb_pred = lgb_model.predict_proba(X)[:, 1]
    xgb_pred = xgb_model.predict_proba(X)[:, 1]

    # catboost needs its own cat handling
    X_cat = X.copy()
    all_features  = cat_model.feature_names_
    cat_indices   = cat_model.get_cat_feature_indices()
    cb_cat_cols   = [all_features[i] for i in cat_indices if all_features[i] in X_cat.columns]
    X_cat[cb_cat_cols] = X_cat[cb_cat_cols].fillna("Missing").astype(str)
    cat_pred = cat_model.predict_proba(X_cat)[:, 1]

    stack_input = pd.DataFrame({"lgb": lgb_pred, "xgb": xgb_pred, "cat": cat_pred})
    final_pred  = stack_model.predict_proba(stack_input)[:, 1]

    os.makedirs("outputs", exist_ok=True)
    submission = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": final_pred})
    submission.to_csv("outputs/predictions.csv", index=False)
    print("Predictions saved to outputs/predictions.csv")


# ------------------------------------------------
# Single prediction (API)
# ------------------------------------------------

def predict_single(app_df, models=None):
    """
    Predict default risk for a single applicant.

    app_df  : DataFrame with one row of raw applicant data
    models  : optional tuple from load_models() — pass in to avoid
              reloading from disk on every API call

    Returns dict with scores and risk label.
    """

    if models is None:
        lgb_model, xgb_model, cat_model, stack_model, feature_cols = load_models()
    else:
        lgb_model, xgb_model, cat_model, stack_model, feature_cols = models

    # --- feature engineering ---
    app_df = preprocess_application(app_df)

    # build a full feature row — start with NaN for everything
    # then fill in what we have from the applicant
    X = pd.DataFrame({col: [np.nan] for col in feature_cols})

    for col in app_df.columns:
        if col in X.columns:
            X[col] = app_df.iloc[0][col]

    # FIX: handle numeric and categorical columns separately
    # don't coerce everything to numeric — that destroys cat cols
    for col in X.columns:
        val = X.iloc[0][col]
        # only coerce if it looks like it should be numeric
        # (i.e. it's not already a proper string category)
        if pd.isna(val):
            continue
        if isinstance(val, bool):
            X[col] = int(val)

    # get the column split the pipeline was trained with
    lgb_preprocessor = lgb_model.named_steps["preprocess"]
    trained_num_cols = [t[2] for t in lgb_preprocessor.transformers if t[0] == "num"][0]
    trained_cat_cols = [t[2] for t in lgb_preprocessor.transformers if t[0] == "cat"][0]

    # cast numeric cols to float so median imputer doesn't choke on object dtype
    for col in trained_num_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

    # cast cat cols to string — keep "Missing" as string, not NaN
    for col in trained_cat_cols:
        if col in X.columns:
            val = X.iloc[0][col]
            X[col] = "Missing" if pd.isna(val) else str(val)

    # replace any inf that crept in from ratio calculations
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

    # --- LightGBM + XGBoost ---
    lgb_pred = lgb_model.predict_proba(X)[:, 1]
    xgb_pred = xgb_model.predict_proba(X)[:, 1]

    # --- CatBoost ---
    X_cat = X.copy()
    all_features = cat_model.feature_names_
    cat_indices  = cat_model.get_cat_feature_indices()
    cb_cat_cols  = [all_features[i] for i in cat_indices if all_features[i] in X_cat.columns]
    X_cat[cb_cat_cols] = X_cat[cb_cat_cols].fillna("Missing").astype(str)
    cat_pred = cat_model.predict_proba(X_cat)[:, 1]

    # --- Stacking ---
    stack_input = pd.DataFrame({
        "lgb": lgb_pred,
        "xgb": xgb_pred,
        "cat": cat_pred
    })
    risk = float(stack_model.predict_proba(stack_input)[0, 1])

    # risk label using tuned threshold
    if risk < 0.2:
        label = "Low Risk"
    elif risk < 0.5:
        label = "Medium Risk"
    else:
        label = "High Risk"

    # build feature vector for DB storage
    feature_vector = {}
    for col in feature_cols:
        val = X.iloc[0][col] if col in X.columns else None
        try:
            feature_vector[col] = float(val) if val is not None and not pd.isna(val) else None
        except (TypeError, ValueError):
            feature_vector[col] = None

    return {
        "stacked":        risk,
        "lgb":            float(lgb_pred[0]),
        "xgb":            float(xgb_pred[0]),
        "cat":            float(cat_pred[0]),
        "risk_label":     label,
        "is_default":     int(risk >= BEST_THRESHOLD),
        "feature_vector": feature_vector,
    }


# ------------------------------------------------
# Run batch prediction
# ------------------------------------------------

if __name__ == "__main__":
    predict()