import os
import gc
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # FIX 4: non-interactive backend so it works without a display
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (
    preprocess_application,
    bureau_features,
    bureau_balance_features,
    previous_application_features,
    pos_features,
    installment_features,
    credit_card_features,
)


def build_sample(n=2000, random_state=42):
    """
    Load only application_train and side tables, build features,
    then return a sample of n rows for SHAP analysis.
    FIX 2: don't load the full 307k rows — sample early to save memory.
    """

    print("Loading application_train...")
    app_train = pd.read_csv("data/application_train.csv")

    # sample early before merging side tables — saves a lot of RAM
    app_sample = app_train.sample(n=min(n * 5, len(app_train)), random_state=random_state)
    del app_train; gc.collect()

    app_sample = preprocess_application(app_sample)

    print("Loading and merging side tables...")

    bureau = pd.read_csv("data/bureau.csv")
    bureau_feat = bureau_features(bureau)
    bureau_bal_feat = bureau_balance_features(bureau, pd.read_csv("data/bureau_balance.csv"))
    app_sample = app_sample.merge(bureau_feat, on="SK_ID_CURR", how="left")
    app_sample = app_sample.merge(bureau_bal_feat, on="SK_ID_CURR", how="left")
    del bureau, bureau_feat, bureau_bal_feat; gc.collect()

    prev_feat = previous_application_features(pd.read_csv("data/previous_application.csv"))
    app_sample = app_sample.merge(prev_feat, on="SK_ID_CURR", how="left")
    del prev_feat; gc.collect()

    pos_feat = pos_features(pd.read_csv("data/POS_CASH_balance.csv"))
    app_sample = app_sample.merge(pos_feat, on="SK_ID_CURR", how="left")
    del pos_feat; gc.collect()

    inst_feat = installment_features(pd.read_csv("data/installments_payments.csv"))
    app_sample = app_sample.merge(inst_feat, on="SK_ID_CURR", how="left")
    del inst_feat; gc.collect()

    cc_feat = credit_card_features(pd.read_csv("data/credit_card_balance.csv"))
    app_sample = app_sample.merge(cc_feat, on="SK_ID_CURR", how="left")
    del cc_feat; gc.collect()

    # FIX 3: drop both TARGET and SK_ID_CURR before transforming
    feature_cols = joblib.load("models/feature_columns.pkl")
    y = app_sample["TARGET"] if "TARGET" in app_sample.columns else None
    X = app_sample.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    X = X[feature_cols]

    # fill cat cols — same as predict pipeline
    cat_cols = X.select_dtypes(include=["object", "str"]).columns
    X[cat_cols] = X[cat_cols].fillna("Missing").astype(str)

    # take final sample
    X = X.sample(n=min(n, len(X)), random_state=random_state).reset_index(drop=True)
    if y is not None:
        y = y.loc[X.index].reset_index(drop=True)

    return X, y


def explain(n_sample=1000, save_dir="outputs/shap"):
    """
    Generate SHAP plots for the LightGBM model.

    n_sample  : number of rows to compute SHAP values on
    save_dir  : directory to save plots
    """

    os.makedirs(save_dir, exist_ok=True)

    print(f"Building sample of {n_sample} rows...")
    X, y = build_sample(n=n_sample)

    print("Loading LightGBM pipeline...")
    lgb_pipeline = joblib.load("models/lgb_model.pkl")
    preprocess   = lgb_pipeline.named_steps["preprocess"]
    model        = lgb_pipeline.named_steps["model"]

    print("Transforming features...")
    feature_names = preprocess.get_feature_names_out()
    X_transformed = pd.DataFrame(
        preprocess.transform(X),
        columns=feature_names
    )

    print("Creating SHAP explainer...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_transformed)

    # -------------------------
    # Plot 1: Summary plot — global feature importance
    # -------------------------
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_transformed, max_display=20, show=False)
    plt.title("SHAP Feature Importance — LightGBM", fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(save_dir, "shap_summary.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path1}")

    # -------------------------
    # Plot 2: Waterfall — single highest-risk prediction
    # -------------------------
    print("Generating SHAP waterfall plot (highest risk applicant)...")
    stacked_scores = lgb_pipeline.predict_proba(X)[:, 1]
    highest_risk_idx = int(np.argmax(stacked_scores))

    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values[highest_risk_idx], max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — Highest Risk Applicant (score={stacked_scores[highest_risk_idx]:.3f})",
        fontsize=13
    )
    plt.tight_layout()
    path2 = os.path.join(save_dir, "shap_waterfall_high_risk.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path2}")

    # -------------------------
    # Plot 3: Waterfall — single lowest-risk prediction
    # -------------------------
    print("Generating SHAP waterfall plot (lowest risk applicant)...")
    lowest_risk_idx = int(np.argmin(stacked_scores))

    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values[lowest_risk_idx], max_display=15, show=False)
    plt.title(
        f"SHAP Waterfall — Lowest Risk Applicant (score={stacked_scores[lowest_risk_idx]:.3f})",
        fontsize=13
    )
    plt.tight_layout()
    path3 = os.path.join(save_dir, "shap_waterfall_low_risk.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path3}")

    # -------------------------
    # Top features summary
    # -------------------------
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features = pd.DataFrame({
        "feature":    feature_names,
        "mean_shap":  mean_abs_shap
    }).sort_values("mean_shap", ascending=False).head(20)

    print("\nTop 20 features by mean |SHAP|:")
    print(top_features.to_string(index=False))

    top_path = os.path.join(save_dir, "top_features.csv")
    top_features.to_csv(top_path, index=False)
    print(f"\nSaved: {top_path}")

    return top_features


if __name__ == "__main__":
    explain()