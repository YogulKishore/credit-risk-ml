import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    log_loss
)


# ----------------------------
# Metric Printer
# ----------------------------

def print_metrics(name, y_true, y_pred):

    preds_binary = (y_pred > 0.5).astype(int)

    print(f"\n{name} Metrics")
    print("-" * 40)

    print("ROC-AUC:", roc_auc_score(y_true, y_pred))
    print("PR-AUC:", average_precision_score(y_true, y_pred))
    print("Log Loss:", log_loss(y_true, y_pred))

    print("\nClassification Metrics")

    print("Accuracy:", accuracy_score(y_true, preds_binary))
    print("Precision:", precision_score(y_true, preds_binary))
    print("Recall:", recall_score(y_true, preds_binary))
    print("F1 Score:", f1_score(y_true, preds_binary))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, preds_binary))


# ----------------------------
# Evaluation
# ----------------------------

def evaluate():

    print("Loading OOF predictions...")

    df = pd.read_csv("models/oof_predictions.csv")

    y = df["TARGET"]

    lgb_pred = df["lgb_oof"]
    xgb_pred = df["xgb_oof"]
    cat_pred = df["cat_oof"]
    stack_pred = df["stack_oof"]

    print("\n==============================")
    print("MODEL EVALUATION REPORT")
    print("==============================")

    print_metrics("LightGBM", y, lgb_pred)
    print_metrics("XGBoost", y, xgb_pred)
    print_metrics("CatBoost", y, cat_pred)
    print_metrics("Stacked Model", y, stack_pred)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    evaluate()