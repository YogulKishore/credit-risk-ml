import pandas as pd
import joblib

from src.features import (
    preprocess_application,
    bureau_features,
    bureau_balance_features,
    previous_application_features,
    pos_features,
    installment_features,
    credit_card_features
)


# ------------------------------------------------
# Load dataset files
# ------------------------------------------------

def load_data():

    app_test = pd.read_csv("data/application_test.csv")

    bureau = pd.read_csv("data/bureau.csv")
    bureau_balance = pd.read_csv("data/bureau_balance.csv")
    prev_app = pd.read_csv("data/previous_application.csv")
    pos = pd.read_csv("data/POS_CASH_balance.csv")
    installments = pd.read_csv("data/installments_payments.csv")
    credit_card = pd.read_csv("data/credit_card_balance.csv")

    return (
        app_test,
        bureau,
        bureau_balance,
        prev_app,
        pos,
        installments,
        credit_card
    )


# ------------------------------------------------
# Feature engineering pipeline
# ------------------------------------------------

def build_features():

    (
        app_test,
        bureau,
        bureau_balance,
        prev_app,
        pos,
        installments,
        credit_card
    ) = load_data()

    app_test = preprocess_application(app_test)

    bureau_feat = bureau_features(bureau)
    app_test = app_test.merge(bureau_feat, on="SK_ID_CURR", how="left")

    bureau_bal_feat = bureau_balance_features(bureau, bureau_balance)
    app_test = app_test.merge(bureau_bal_feat, on="SK_ID_CURR", how="left")

    prev_feat = previous_application_features(prev_app)
    app_test = app_test.merge(prev_feat, on="SK_ID_CURR", how="left")

    pos_feat = pos_features(pos)
    app_test = app_test.merge(pos_feat, on="SK_ID_CURR", how="left")

    inst_feat = installment_features(installments)
    app_test = app_test.merge(inst_feat, on="SK_ID_CURR", how="left")

    cc_feat = credit_card_features(credit_card)
    app_test = app_test.merge(cc_feat, on="SK_ID_CURR", how="left")

    return app_test


# ------------------------------------------------
# Batch prediction (Kaggle)
# ------------------------------------------------

def predict():

    print("Loading models...")

    lgb_model = joblib.load("models/lgb_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    cat_model = joblib.load("models/cat_model.pkl")
    stack_model = joblib.load("models/stack_model.pkl")

    feature_cols = joblib.load("models/feature_columns.pkl")

    print("Building features...")

    data = build_features()

    ids = data["SK_ID_CURR"]

    X = data.drop(columns=["SK_ID_CURR"])

    X = X[feature_cols]

    cat_cols = X.select_dtypes("object").columns

    X[cat_cols] = X[cat_cols].fillna("Missing").astype(str)

    print("Generating base model predictions...")

    lgb_pred = lgb_model.predict_proba(X)[:,1]

    xgb_pred = xgb_model.predict_proba(X)[:,1]

    cat_pred = cat_model.predict_proba(X)[:,1]

    print("Applying stacking model...")

    stack_input = pd.DataFrame({
        "lgb": lgb_pred,
        "xgb": xgb_pred,
        "cat": cat_pred
    })

    final_pred = stack_model.predict_proba(stack_input)[:,1]

    submission = pd.DataFrame({
        "SK_ID_CURR": ids,
        "TARGET": final_pred
    })

    submission.to_csv("outputs/predictions.csv", index=False)

    print("Predictions saved to outputs/predictions.csv")


# ------------------------------------------------
# Single prediction (Streamlit)
# ------------------------------------------------

def predict_lgb_xgb(X):

    import joblib

    lgb_model = joblib.load("models/lgb_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")

    lgb_pred = lgb_model.predict_proba(X)[:,1]
    xgb_pred = xgb_model.predict_proba(X)[:,1]

    return lgb_pred, xgb_pred

def predict_catboost(X):

    import joblib

    cat_model = joblib.load("models/cat_model.pkl")

    X_cat = X.copy()

    # Get cat column names from model's feature list using cat indices
    all_features = cat_model.feature_names_
    cat_indices = cat_model.get_cat_feature_indices()
    cat_cols = [all_features[i] for i in cat_indices]
    cat_cols = [col for col in cat_cols if col in X_cat.columns]

    X_cat[cat_cols] = X_cat[cat_cols].fillna("Missing").astype(str)

    cat_pred = cat_model.predict_proba(X_cat)[:,1]

    return cat_pred

def predict_single(app_df):

    import pandas as pd
    import numpy as np
    import joblib

    stack_model = joblib.load("models/stack_model.pkl")

    feature_cols = joblib.load("models/feature_columns.pkl")

    app_df = preprocess_application(app_df)

    X = pd.DataFrame({col: [np.nan] for col in feature_cols})

    # cast all columns to object first to allow mixed types
    X = X.astype(object)
    
    for col in app_df.columns:
        if col in X.columns:
            X.loc[0, col] = app_df.iloc[0][col]

    X = X.replace({True: 1, False: 0})

    # Coerce any stray object/string columns to numeric so the median imputer doesn't choke
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # -------------------
    # LightGBM + XGBoost
    # -------------------

    lgb_pred, xgb_pred = predict_lgb_xgb(X)

    # -------------------
    # CatBoost
    # -------------------

    cat_pred = predict_catboost(X)

    # -------------------
    # Stacking
    # -------------------

    stack_input = pd.DataFrame({
        "lgb": lgb_pred,
        "xgb": xgb_pred,
        "cat": cat_pred
    })

    risk = stack_model.predict_proba(stack_input)[0,1]

    # build feature vector dict for storage
    feature_vector = {col: (float(X.iloc[0][col]) if col in X.columns else None) for col in feature_cols}

    return {
        "stacked": float(risk),
        "lgb": float(lgb_pred[0]),
        "xgb": float(xgb_pred[0]),
        "cat": float(cat_pred[0]),
        "feature_vector": feature_vector,
    }

# ------------------------------------------------
# Run batch prediction
# ------------------------------------------------

if __name__ == "__main__":
    predict()