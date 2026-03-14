import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from features import (
    preprocess_application,
    bureau_features,
    bureau_balance_features,
    previous_application_features,
    pos_features,
    installment_features,
    credit_card_features
)


# ----------------------------
# Load Data
# ----------------------------

def load_data():

    app_train = pd.read_csv("data/application_train.csv")
    app_test = pd.read_csv("data/application_test.csv")

    bureau = pd.read_csv("data/bureau.csv")
    bureau_balance = pd.read_csv("data/bureau_balance.csv")
    prev_app = pd.read_csv("data/previous_application.csv")
    pos = pd.read_csv("data/POS_CASH_balance.csv")
    installments = pd.read_csv("data/installments_payments.csv")
    credit_card = pd.read_csv("data/credit_card_balance.csv")

    return (
        app_train,
        app_test,
        bureau,
        bureau_balance,
        prev_app,
        pos,
        installments,
        credit_card
    )


# ----------------------------
# Feature Engineering
# ----------------------------

def build_features():

    (
        app_train,
        app_test,
        bureau,
        bureau_balance,
        prev_app,
        pos,
        installments,
        credit_card
    ) = load_data()

    app_train = preprocess_application(app_train)
    app_test = preprocess_application(app_test)

    bureau_feat = bureau_features(bureau)
    app_train = app_train.merge(bureau_feat, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(bureau_feat, on="SK_ID_CURR", how="left")

    bureau_bal_feat = bureau_balance_features(bureau, bureau_balance)
    app_train = app_train.merge(bureau_bal_feat, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(bureau_bal_feat, on="SK_ID_CURR", how="left")

    prev_feat = previous_application_features(prev_app)
    app_train = app_train.merge(prev_feat, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(prev_feat, on="SK_ID_CURR", how="left")

    pos_feat = pos_features(pos)
    app_train = app_train.merge(pos_feat, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(pos_feat, on="SK_ID_CURR", how="left")

    inst_feat = installment_features(installments)
    app_train = app_train.merge(inst_feat, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(inst_feat, on="SK_ID_CURR", how="left")

    cc_feat = credit_card_features(credit_card)
    app_train = app_train.merge(cc_feat, on="SK_ID_CURR", how="left")
    app_test = app_test.merge(cc_feat, on="SK_ID_CURR", how="left")

    return app_train, app_test


# ----------------------------
# Preprocessing
# ----------------------------

def build_preprocessor(X):

    num_cols = X.select_dtypes(exclude="object").columns
    cat_cols = X.select_dtypes(include="object").columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    return preprocessor


# ----------------------------
# Train Models
# ----------------------------

def train_models(X, y, X_test, test_ids):

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lgb_preds = np.zeros(len(X))
    xgb_preds = np.zeros(len(X))
    cat_preds = np.zeros(len(X))

    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    preprocessor = build_preprocessor(X)

    # LIGHTGBM
    lgb_model = Pipeline([
        ("preprocess", preprocessor),
        ("model", lgb.LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
            n_jobs = -1
))
    ])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
        # preprocess
        lgb_model.named_steps["preprocess"].fit(X_train)
    
        X_train_p = lgb_model.named_steps["preprocess"].transform(X_train)
        X_val_p = lgb_model.named_steps["preprocess"].transform(X_val)
        X_test_p = lgb_model.named_steps["preprocess"].transform(X_test)
    
        # train
        lgb_model.named_steps["model"].fit(
            X_train_p,
            y_train,
            eval_set=[(X_val_p, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100)]
        )
    
        preds = lgb_model.named_steps["model"].predict_proba(X_val_p)[:,1]
    
        lgb_preds[val_idx] = preds
    
        test_lgb += (
            lgb_model.named_steps["model"].predict_proba(X_test_p)[:,1]
            / skf.n_splits
        )

    print("LightGBM ROC-AUC:", roc_auc_score(y, lgb_preds))


    # XGBOOST
    xgb_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", xgb.XGBClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="hist",
    device="cuda",
    random_state=42,
    early_stopping_rounds=100
))
    ])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
        xgb_pipeline.named_steps["preprocess"].fit(X_train)
    
        X_train_p = xgb_pipeline.named_steps["preprocess"].transform(X_train)
        X_val_p = xgb_pipeline.named_steps["preprocess"].transform(X_val)
        X_test_p = xgb_pipeline.named_steps["preprocess"].transform(X_test)
    
        xgb_pipeline.named_steps["model"].fit(
            X_train_p,
            y_train,
            eval_set=[(X_val_p, y_val)],
            verbose=200
        )
    
        preds = xgb_pipeline.named_steps["model"].predict_proba(X_val_p)[:,1]
    
        xgb_preds[val_idx] = preds
    
        test_xgb += (
            xgb_pipeline.named_steps["model"].predict_proba(X_test_p)[:,1]
            / skf.n_splits
        )
    
    print("XGBoost ROC-AUC:", roc_auc_score(y, xgb_preds))


    # CATBOOST
    cat_cols = X.select_dtypes("object").columns.tolist()
    
    X[cat_cols] = X[cat_cols].fillna("Missing").astype(str)
    X_test[cat_cols] = X_test[cat_cols].fillna("Missing").astype(str)
    
    cat_model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    random_state=42,
    task_type="GPU",
    devices="0",
    od_type="Iter",
    od_wait=100,
    verbose=200
)

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cat_model.fit(
            X_train,
            y_train,
            cat_features=[X.columns.get_loc(c) for c in cat_cols],
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose = 200
        )

        preds = cat_model.predict_proba(X_val)[:,1]
        cat_preds[val_idx] = preds

        test_cat += cat_model.predict_proba(X_test)[:,1] / skf.n_splits

    print("CatBoost ROC-AUC:", roc_auc_score(y, cat_preds))


    # STACKING
    stack_train = pd.DataFrame({
        "lgb": lgb_preds,
        "xgb": xgb_preds,
        "cat": cat_preds
    })

    stack_test = pd.DataFrame({
        "lgb": test_lgb,
        "xgb": test_xgb,
        "cat": test_cat
    })

    meta_model = LogisticRegression()
    meta_model.fit(stack_train, y)

    final_preds = meta_model.predict_proba(stack_train)[:,1]

    print("Stacked ROC-AUC:", roc_auc_score(y, final_preds))

    oof_df = pd.DataFrame({
    "TARGET": y,
    "lgb_oof": lgb_preds,
    "xgb_oof": xgb_preds,
    "cat_oof": cat_preds,
    "stack_oof": final_preds
})

    oof_df.to_csv("models/oof_predictions.csv", index=False)

    # ----------------------------
    # MLflow Logging
    # ----------------------------

    mlflow.set_experiment("credit-risk")

    with mlflow.start_run(run_name="stacked_ensemble"):

        # log metrics
        mlflow.log_metric("lgb_roc_auc",    roc_auc_score(y, lgb_preds))
        mlflow.log_metric("xgb_roc_auc",    roc_auc_score(y, xgb_preds))
        mlflow.log_metric("cat_roc_auc",    roc_auc_score(y, cat_preds))
        mlflow.log_metric("stacked_roc_auc", roc_auc_score(y, final_preds))

        # log params
        mlflow.log_param("n_folds", 5)
        mlflow.log_param("lgb_n_estimators", 5000)
        mlflow.log_param("lgb_learning_rate", 0.05)
        mlflow.log_param("lgb_num_leaves", 64)
        mlflow.log_param("xgb_n_estimators", 5000)
        mlflow.log_param("xgb_learning_rate", 0.05)
        mlflow.log_param("xgb_max_depth", 6)
        mlflow.log_param("cat_iterations", 5000)
        mlflow.log_param("cat_learning_rate", 0.05)
        mlflow.log_param("cat_depth", 6)
        mlflow.log_param("meta_model", "LogisticRegression")

        # log models
        mlflow.sklearn.log_model(meta_model, "stack_model")
        mlflow.lightgbm.log_model(lgb_model.named_steps["model"], "lgb_model")
        mlflow.xgboost.log_model(xgb_pipeline.named_steps["model"], "xgb_model")

        # log feature columns artifact
        mlflow.log_artifact("models/feature_columns.pkl")
        mlflow.log_artifact("models/oof_predictions.csv")

        print("MLflow run logged.")

    # SAVE MODELS
    # SAVE FEATURE ORDER
    joblib.dump(X.columns.tolist(), "C:/Users/myself/Projects/machine_learning/machine_learning_1/models/feature_columns.pkl")
    
    # SAVE MODELS
    joblib.dump(lgb_model, "C:/Users/myself/Projects/machine_learning/machine_learning_1/models/lgb_model.pkl")
    joblib.dump(xgb_pipeline, "C:/Users/myself/Projects/machine_learning/machine_learning_1/models/xgb_model.pkl")
    joblib.dump(cat_model, "C:/Users/myself/Projects/machine_learning/machine_learning_1/models/cat_model.pkl")
    joblib.dump(meta_model, "C:/Users/myself/Projects/machine_learning/machine_learning_1/models/stack_model.pkl")
    
    print("Models and feature columns saved.")

    # Final test predictions

    test_stack = pd.DataFrame({
        "lgb": test_lgb,
        "xgb": test_xgb,
        "cat": test_cat
    })
    
    test_final = meta_model.predict_proba(test_stack)[:,1]
    
    submission = pd.DataFrame({
    "SK_ID_CURR": test_ids,
    "TARGET": test_final
})
    
    submission.to_csv("outputs/test_predictions.csv", index=False)
    
    print("Test predictions saved.")

# ----------------------------
# MAIN
# ----------------------------

def main():

    app_train, app_test = build_features()

    y = app_train["TARGET"]
    X = app_train.drop(columns=["TARGET", "SK_ID_CURR"])
    X_test = app_test.drop(columns=["SK_ID_CURR"])

    train_models(X, y, X_test, app_test["SK_ID_CURR"])


if __name__ == "__main__":
    main()