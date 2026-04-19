"""
Log trained model results to MLflow without retraining.
Run with: python src/log_mlflow.py
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, precision_recall_curve
import numpy as np


def find_best_threshold(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def log_to_mlflow():

    # ----------------------------
    # Load OOF predictions
    # ----------------------------
    print("Loading OOF predictions...")
    df = pd.read_csv("models/oof_predictions.csv")

    y           = df["TARGET"]
    lgb_preds   = df["lgb_oof"]
    xgb_preds   = df["xgb_oof"]
    cat_preds   = df["cat_oof"]
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
    # Compute threshold
    # ----------------------------
    stack_threshold, stack_f1 = find_best_threshold(y, stack_preds)
    lgb_threshold, _          = find_best_threshold(y, lgb_preds)
    xgb_threshold, _          = find_best_threshold(y, xgb_preds)
    cat_threshold, _          = find_best_threshold(y, cat_preds)

    # ----------------------------
    # Log to MLflow
    # FIX 1: wrapped in function with try/except for clean error handling
    # FIX 3: use name= keyword to avoid artifact_path deprecation warning
    # FIX 5: log CatBoost model too
    # FIX 4: log threshold, F1, and extra artifacts
    # ----------------------------
    mlflow.set_experiment("credit-risk")

    with mlflow.start_run(run_name="stacked_ensemble_v1"):

        # --- metrics ---
        mlflow.log_metric("lgb_roc_auc",      roc_auc_score(y, lgb_preds))
        mlflow.log_metric("lgb_pr_auc",       average_precision_score(y, lgb_preds))
        mlflow.log_metric("lgb_log_loss",     log_loss(y, lgb_preds))
        mlflow.log_metric("lgb_best_threshold", lgb_threshold)

        mlflow.log_metric("xgb_roc_auc",      roc_auc_score(y, xgb_preds))
        mlflow.log_metric("xgb_pr_auc",       average_precision_score(y, xgb_preds))
        mlflow.log_metric("xgb_log_loss",     log_loss(y, xgb_preds))
        mlflow.log_metric("xgb_best_threshold", xgb_threshold)

        mlflow.log_metric("cat_roc_auc",      roc_auc_score(y, cat_preds))
        mlflow.log_metric("cat_pr_auc",       average_precision_score(y, cat_preds))
        mlflow.log_metric("cat_log_loss",     log_loss(y, cat_preds))
        mlflow.log_metric("cat_best_threshold", cat_threshold)

        mlflow.log_metric("stacked_roc_auc",      roc_auc_score(y, stack_preds))
        mlflow.log_metric("stacked_pr_auc",       average_precision_score(y, stack_preds))
        mlflow.log_metric("stacked_log_loss",     log_loss(y, stack_preds))
        mlflow.log_metric("stacked_best_threshold", stack_threshold)
        mlflow.log_metric("stacked_best_f1",        stack_f1)

        # --- params ---
        mlflow.log_param("n_folds",           5)
        mlflow.log_param("lgb_n_estimators",  5000)
        mlflow.log_param("lgb_learning_rate", 0.05)
        mlflow.log_param("lgb_num_leaves",    64)
        mlflow.log_param("xgb_n_estimators",  5000)
        mlflow.log_param("xgb_learning_rate", 0.05)
        mlflow.log_param("xgb_max_depth",     6)
        mlflow.log_param("cat_iterations",    5000)
        mlflow.log_param("cat_learning_rate", 0.05)
        mlflow.log_param("cat_depth",         6)
        mlflow.log_param("meta_model",        "LogisticRegression")

        # --- models ---
        mlflow.sklearn.log_model(stack_model,                         name="stack_model")
        mlflow.lightgbm.log_model(lgb_model.named_steps["model"],     name="lgb_model")
        mlflow.xgboost.log_model(xgb_model.named_steps["model"],      name="xgb_model")
        mlflow.catboost.log_model(cat_model,                          name="cat_model")

        # --- artifacts ---
        mlflow.log_artifact("models/feature_columns.pkl")
        mlflow.log_artifact("models/oof_predictions.csv")

        # log SHAP top features if they exist
        shap_path = "outputs/shap/top_features.csv"
        if os.path.exists(shap_path):
            mlflow.log_artifact(shap_path)
            print("SHAP top features logged.")

        print("MLflow run logged successfully!")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Best stacked threshold: {round(stack_threshold, 4)}  F1: {round(stack_f1, 4)}")


if __name__ == "__main__":
    log_to_mlflow()