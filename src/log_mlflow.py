"""
Log existing trained model results to MLflow without retraining.
Run with: python scripts/log_mlflow.py
"""

import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

# ----------------------------
# Load OOF predictions
# ----------------------------

print("Loading OOF predictions...")
df = pd.read_csv("models/oof_predictions.csv")

y         = df["TARGET"]
lgb_preds  = df["lgb_oof"]
xgb_preds  = df["xgb_oof"]
cat_preds  = df["cat_oof"]
stack_preds = df["stack_oof"]

# ----------------------------
# Load models
# ----------------------------

print("Loading models...")
lgb_model   = joblib.load("models/lgb_model.pkl")
xgb_model   = joblib.load("models/xgb_model.pkl")
cat_model   = joblib.load("models/cat_model.pkl")
stack_model = joblib.load("models/stack_model.pkl")

# ----------------------------
# Log to MLflow
# ----------------------------

mlflow.set_experiment("credit-risk")

with mlflow.start_run(run_name="stacked_ensemble_v1"):

    # metrics
    mlflow.log_metric("lgb_roc_auc",     roc_auc_score(y, lgb_preds))
    mlflow.log_metric("lgb_pr_auc",      average_precision_score(y, lgb_preds))
    mlflow.log_metric("lgb_log_loss",    log_loss(y, lgb_preds))

    mlflow.log_metric("xgb_roc_auc",     roc_auc_score(y, xgb_preds))
    mlflow.log_metric("xgb_pr_auc",      average_precision_score(y, xgb_preds))
    mlflow.log_metric("xgb_log_loss",    log_loss(y, xgb_preds))

    mlflow.log_metric("cat_roc_auc",     roc_auc_score(y, cat_preds))
    mlflow.log_metric("cat_pr_auc",      average_precision_score(y, cat_preds))
    mlflow.log_metric("cat_log_loss",    log_loss(y, cat_preds))

    mlflow.log_metric("stacked_roc_auc", roc_auc_score(y, stack_preds))
    mlflow.log_metric("stacked_pr_auc",  average_precision_score(y, stack_preds))
    mlflow.log_metric("stacked_log_loss", log_loss(y, stack_preds))

    # params
    mlflow.log_param("n_folds",            5)
    mlflow.log_param("lgb_n_estimators",   5000)
    mlflow.log_param("lgb_learning_rate",  0.05)
    mlflow.log_param("lgb_num_leaves",     64)
    mlflow.log_param("xgb_n_estimators",   5000)
    mlflow.log_param("xgb_learning_rate",  0.05)
    mlflow.log_param("xgb_max_depth",      6)
    mlflow.log_param("cat_iterations",     5000)
    mlflow.log_param("cat_learning_rate",  0.05)
    mlflow.log_param("cat_depth",          6)
    mlflow.log_param("meta_model",         "LogisticRegression")

    # models
    mlflow.sklearn.log_model(stack_model, "stack_model")
    mlflow.lightgbm.log_model(lgb_model.named_steps["model"], "lgb_model")
    mlflow.xgboost.log_model(xgb_model.named_steps["model"], "xgb_model")

    # artifacts
    mlflow.log_artifact("models/feature_columns.pkl")

    print("MLflow run logged successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")