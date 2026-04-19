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

def print_metrics(name, y_true, y_pred, threshold=0.5):

    preds_binary = (y_pred > threshold).astype(int)

    print(f"\n{name} Metrics (threshold={threshold})")
    print("-" * 50)

    print("ROC-AUC:  ", round(roc_auc_score(y_true, y_pred), 4))
    print("PR-AUC:   ", round(average_precision_score(y_true, y_pred), 4))
    print("Log Loss: ", round(log_loss(y_true, y_pred), 4))
    print("Accuracy: ", round(accuracy_score(y_true, preds_binary), 4))
    print("Precision:", round(precision_score(y_true, preds_binary, zero_division=0), 4))
    print("Recall:   ", round(recall_score(y_true, preds_binary, zero_division=0), 4))
    print("F1 Score: ", round(f1_score(y_true, preds_binary, zero_division=0), 4))

    cm = confusion_matrix(y_true, preds_binary)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (correct non-default): {tn:>7}")
    print(f"  False Positives (flagged wrongly):      {fp:>7}")
    print(f"  False Negatives (missed defaults):      {fn:>7}  ← expensive")
    print(f"  True Positives  (caught defaults):      {tp:>7}")


# ----------------------------
# Threshold Optimization
# FIX: return the best threshold so predict.py can use it
# FIX: run on any model, not just stacked
# ----------------------------

def find_best_threshold(y_true, y_pred, name="Model"):

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\n{name} — Best Threshold: {round(best_threshold, 4)}  (F1={round(best_f1, 4)})")

    print_metrics(name, y_true, y_pred, threshold=best_threshold)

    return float(best_threshold)


# ----------------------------
# Evaluation
# ----------------------------

def evaluate():

    print("Loading OOF predictions...")

    df = pd.read_csv("models/oof_predictions.csv")

    y          = df["TARGET"]
    lgb_pred   = df["lgb_oof"]
    xgb_pred   = df["xgb_oof"]
    cat_pred   = df["cat_oof"]
    stack_pred = df["stack_oof"]

    print("\n==============================")
    print("MODEL EVALUATION REPORT (threshold=0.5)")
    print("==============================")

    print_metrics("LightGBM",     y, lgb_pred)
    print_metrics("XGBoost",      y, xgb_pred)
    print_metrics("CatBoost",     y, cat_pred)
    print_metrics("Stacked Model",y, stack_pred)

    print("\n==============================")
    print("THRESHOLD TUNING (all models)")
    print("==============================")

    lgb_thresh   = find_best_threshold(y, lgb_pred,   "LightGBM")
    xgb_thresh   = find_best_threshold(y, xgb_pred,   "XGBoost")
    cat_thresh   = find_best_threshold(y, cat_pred,   "CatBoost")
    stack_thresh = find_best_threshold(y, stack_pred, "Stacked Model")

    print("\n==============================")
    print("THRESHOLD SUMMARY")
    print("==============================")
    print(f"  LightGBM best threshold:     {round(lgb_thresh, 4)}")
    print(f"  XGBoost  best threshold:     {round(xgb_thresh, 4)}")
    print(f"  CatBoost best threshold:     {round(cat_thresh, 4)}")
    print(f"  Stacked  best threshold:     {round(stack_thresh, 4)}  ← use this in predict.py")

    return stack_thresh


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    evaluate()