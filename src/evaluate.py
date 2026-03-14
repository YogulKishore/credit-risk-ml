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
    log_loss,
    precision_recall_curve
)


# ----------------------------
# Metric Printer
# ----------------------------

def print_metrics(name, y_true, y_pred):

    preds_binary = (y_pred > 0.5).astype(int)

    print(f"\n{name} Metrics")
    print("-" * 50)

    print("ROC-AUC:", roc_auc_score(y_true, y_pred))
    print("PR-AUC:", average_precision_score(y_true, y_pred))
    print("Log Loss:", log_loss(y_true, y_pred))

    print("\nClassification Metrics (threshold = 0.5)")

    print("Accuracy:", accuracy_score(y_true, preds_binary))
    print("Precision:", precision_score(y_true, preds_binary))
    print("Recall:", recall_score(y_true, preds_binary))
    print("F1 Score:", f1_score(y_true, preds_binary))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, preds_binary))


# ----------------------------
# Threshold Optimization
# ----------------------------

def find_best_threshold(y_true, y_pred):

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]

    print("\nBest Threshold (F1 Optimization):", best_threshold)

    preds_binary = (y_pred > best_threshold).astype(int)

    print("\nMetrics using optimized threshold")

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

    print("\n==============================")
    print("STACKED MODEL THRESHOLD TUNING")
    print("==============================")

    find_best_threshold(y, stack_pred)


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    evaluate()