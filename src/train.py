import os, gc, subprocess
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import mlflow, mlflow.sklearn, mlflow.lightgbm, mlflow.xgboost

from features import (
    preprocess_application, bureau_features, bureau_balance_features,
    previous_application_features, pos_features,
    installment_features, credit_card_features
)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def reduce_memory(df):
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = df[col].astype("int32")
    return df


def get_device():
    try:
        if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
            return "cuda", "GPU"
    except FileNotFoundError:
        pass
    return "cpu", "CPU"


def build_preprocessor(X):
    num_cols = X.select_dtypes(exclude=["object", "str"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median"))
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ])

# -------------------------------------------------------
# Data loading & feature engineering
# -------------------------------------------------------

def build_features():
    print("Loading CSVs...")
    app_train      = reduce_memory(pd.read_csv("data/application_train.csv"))
    app_test       = reduce_memory(pd.read_csv("data/application_test.csv"))
    bureau         = reduce_memory(pd.read_csv("data/bureau.csv"))
    bureau_balance = reduce_memory(pd.read_csv("data/bureau_balance.csv"))
    prev_app       = reduce_memory(pd.read_csv("data/previous_application.csv"))
    pos            = reduce_memory(pd.read_csv("data/POS_CASH_balance.csv"))
    installments   = reduce_memory(pd.read_csv("data/installments_payments.csv"))
    credit_card    = reduce_memory(pd.read_csv("data/credit_card_balance.csv"))

    print("Engineering features...")
    app_train = preprocess_application(app_train)
    app_test  = preprocess_application(app_test)

    bureau_feat = bureau_features(bureau)
    app_train = app_train.merge(bureau_feat, on="SK_ID_CURR", how="left")
    app_test  = app_test.merge(bureau_feat,  on="SK_ID_CURR", how="left")
    del bureau_feat; gc.collect()

    bureau_bal_feat = bureau_balance_features(bureau, bureau_balance)
    app_train = app_train.merge(bureau_bal_feat, on="SK_ID_CURR", how="left")
    app_test  = app_test.merge(bureau_bal_feat,  on="SK_ID_CURR", how="left")
    del bureau, bureau_balance, bureau_bal_feat; gc.collect()

    prev_feat = previous_application_features(prev_app)
    app_train = app_train.merge(prev_feat, on="SK_ID_CURR", how="left")
    app_test  = app_test.merge(prev_feat,  on="SK_ID_CURR", how="left")
    del prev_app, prev_feat; gc.collect()

    pos_feat = pos_features(pos)
    app_train = app_train.merge(pos_feat, on="SK_ID_CURR", how="left")
    app_test  = app_test.merge(pos_feat,  on="SK_ID_CURR", how="left")
    del pos, pos_feat; gc.collect()

    inst_feat = installment_features(installments)
    app_train = app_train.merge(inst_feat, on="SK_ID_CURR", how="left")
    app_test  = app_test.merge(inst_feat,  on="SK_ID_CURR", how="left")
    del installments, inst_feat; gc.collect()

    cc_feat = credit_card_features(credit_card)
    app_train = app_train.merge(cc_feat, on="SK_ID_CURR", how="left")
    app_test  = app_test.merge(cc_feat,  on="SK_ID_CURR", how="left")
    del credit_card, cc_feat; gc.collect()

    return app_train, app_test

# -------------------------------------------------------
# Training
# -------------------------------------------------------

def train_models(X, y, X_test, test_ids):

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # FIX 3: snapshot raw data for CatBoost BEFORE any filling
    # CatBoost handles NaNs natively — don't pre-fill or it loses that signal
    X_cat_raw      = X.copy()
    X_test_cat_raw = X_test.copy()

    # FIX 1: fill cat cols only for lgb/xgb (they need string, not NaN)
    # the pipeline's SimpleImputer would also do this, but filling upfront
    # avoids mixed-type warnings from pandas when iloc slicing inside the loop
    cat_cols_all = X.select_dtypes(include=["object", "str"]).columns.tolist()
    X[cat_cols_all]      = X[cat_cols_all].fillna("Missing").astype(str)
    X_test[cat_cols_all] = X_test[cat_cols_all].fillna("Missing").astype(str)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgb_preds = np.zeros(len(X))
    xgb_preds = np.zeros(len(X))
    cat_preds = np.zeros(len(X))
    test_lgb  = np.zeros(len(X_test))
    test_xgb  = np.zeros(len(X_test))
    test_cat  = np.zeros(len(X_test))
    xgb_device, cat_device = get_device()
    print(f"Device: {xgb_device.upper()}")

    # ---- LightGBM ----
    print("\n--- LightGBM ---")
    lgb_best_iters = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/5")
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        pre = build_preprocessor(X_tr)
        pre.fit(X_tr)
        Xtr_p = pre.transform(X_tr)
        Xva_p = pre.transform(X_va)
        Xte_p = pre.transform(X_test)

        m = lgb.LGBMClassifier(
            n_estimators=5000, learning_rate=0.05, max_depth=-1,
            num_leaves=64, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        m.fit(Xtr_p, y_tr, eval_set=[(Xva_p, y_va)],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])

        lgb_preds[va] = m.predict_proba(Xva_p)[:, 1]
        test_lgb     += m.predict_proba(Xte_p)[:, 1] / 5
        lgb_best_iters.append(m.best_iteration_)

    print("LGB OOF AUC:", round(roc_auc_score(y, lgb_preds), 4))

    # ---- XGBoost ----
    print("\n--- XGBoost ---")
    xgb_best_iters = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/5")
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        pre = build_preprocessor(X_tr)
        pre.fit(X_tr)
        Xtr_p = pre.transform(X_tr)
        Xva_p = pre.transform(X_va)
        Xte_p = pre.transform(X_test)

        m = xgb.XGBClassifier(
            n_estimators=5000, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="auc", tree_method="hist",
            device=xgb_device, random_state=42,
            early_stopping_rounds=100, verbosity=0
        )
        m.fit(Xtr_p, y_tr, eval_set=[(Xva_p, y_va)], verbose=500)

        xgb_preds[va] = m.predict_proba(Xva_p)[:, 1]
        test_xgb     += m.predict_proba(Xte_p)[:, 1] / 5
        xgb_best_iters.append(m.best_iteration)

    print("XGB OOF AUC:", round(roc_auc_score(y, xgb_preds), 4))
    gc.collect()

    # ---- CatBoost ----
    print("\n--- CatBoost ---")
    # FIX 3: use raw copy (NaNs intact) — CatBoost handles missingness natively
    X_cat      = X_cat_raw
    X_test_cat = X_test_cat_raw
    cat_cols   = X_cat.select_dtypes(include=["object", "str"]).columns.tolist()
    cat_best_iters = []

    for fold, (tr, va) in enumerate(skf.split(X_cat, y)):
        print(f"  Fold {fold+1}/5")
        X_tr, X_va = X_cat.iloc[tr], X_cat.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        m = CatBoostClassifier(
            iterations=5000, learning_rate=0.05, depth=6,
            eval_metric="AUC", random_seed=42,
            task_type=cat_device, od_type="Iter", od_wait=100, verbose=500
        )
        m.fit(X_tr, y_tr,
              cat_features=[X_cat.columns.get_loc(c) for c in cat_cols],
              eval_set=(X_va, y_va))

        cat_preds[va] = m.predict_proba(X_va)[:, 1]
        test_cat     += m.predict_proba(X_test_cat)[:, 1] / 5
        cat_best_iters.append(m.best_iteration_)

    cat_model = m
    print("Cat OOF AUC:", round(roc_auc_score(y, cat_preds), 4))
    gc.collect()

    # ---- Stacking ----
    # FIX 4: evaluate meta-learner honestly with its own CV
    # base model OOF preds are already leak-free; meta must be too
    print("\n--- Stacking ---")
    stack_train = pd.DataFrame({"lgb": lgb_preds, "xgb": xgb_preds, "cat": cat_preds})
    stack_test  = pd.DataFrame({"lgb": test_lgb,  "xgb": test_xgb,  "cat": test_cat})

    # honest OOF evaluation of meta-learner using its own 5-fold CV
    meta_oof = np.zeros(len(X))
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(stack_train, y):
        meta_cv = LogisticRegression(max_iter=1000, random_state=42)
        meta_cv.fit(stack_train.iloc[tr], y.iloc[tr])
        meta_oof[va] = meta_cv.predict_proba(stack_train.iloc[va])[:, 1]
    print("Stacked OOF AUC (honest):", round(roc_auc_score(y, meta_oof), 4))

    # final meta-learner trained on ALL base OOF preds for saving/inference
    meta = LogisticRegression(max_iter=1000, random_state=42)
    meta.fit(stack_train, y)

    # ---- FIX 2: refit all models on full data before saving ----
    # CV folds trained on 80% — saved model should know 100% of training data
    print("\nRefitting on full data for saving...")

    print("  Refitting LightGBM...")
    pre_lgb = build_preprocessor(X)
    pre_lgb.fit(X)
    X_full_lgb = pre_lgb.transform(X)
    lgb_final = lgb.LGBMClassifier(
        n_estimators=int(np.mean(lgb_best_iters)),  # avg best iter across folds
        learning_rate=0.05, max_depth=-1,
        num_leaves=64, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_final.fit(X_full_lgb, y)
    lgb_pipeline = Pipeline([("preprocess", pre_lgb), ("model", lgb_final)])

    print("  Refitting XGBoost...")
    pre_xgb = build_preprocessor(X)
    pre_xgb.fit(X)
    X_full_xgb = pre_xgb.transform(X)
    xgb_final = xgb.XGBClassifier(
        n_estimators=int(np.mean(xgb_best_iters)),
        learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="auc", tree_method="hist",
        device=xgb_device, random_state=42, verbosity=0
    )
    xgb_final.fit(X_full_xgb, y)
    xgb_pipeline = Pipeline([("preprocess", pre_xgb), ("model", xgb_final)])

    print("  Refitting CatBoost...")
    cat_final = CatBoostClassifier(
        iterations=int(np.mean(cat_best_iters)),
        learning_rate=0.05, depth=6,
        eval_metric="AUC", random_seed=42,
        task_type=cat_device, verbose=0
    )
    cat_final.fit(
        X_cat, y,
        cat_features=[X_cat.columns.get_loc(c) for c in cat_cols]
    )
    cat_model = cat_final

    # ---- Save ----
    pd.DataFrame({
        "TARGET": y.values, "lgb_oof": lgb_preds,
        "xgb_oof": xgb_preds, "cat_oof": cat_preds, "stack_oof": meta_oof
    }).to_csv("models/oof_predictions.csv", index=False)

    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    joblib.dump(lgb_pipeline,       "models/lgb_model.pkl")
    joblib.dump(xgb_pipeline,       "models/xgb_model.pkl")
    joblib.dump(cat_model,          "models/cat_model.pkl")
    joblib.dump(meta,               "models/stack_model.pkl")
    print("Models saved.")

    # ---- MLflow ----
    try:
        mlflow.set_experiment("credit-risk")
        with mlflow.start_run(run_name="stacked_ensemble"):
            mlflow.log_metrics({
                "lgb_roc_auc":     roc_auc_score(y, lgb_preds),
                "xgb_roc_auc":     roc_auc_score(y, xgb_preds),
                "cat_roc_auc":     roc_auc_score(y, cat_preds),
                "stacked_roc_auc": roc_auc_score(y, meta_oof),
            })
            mlflow.log_params({
                "n_folds": 5, "lgb_num_leaves": 64,
                "xgb_max_depth": 6, "cat_depth": 6,
                "meta_model": "LogisticRegression", "device": xgb_device
            })
            mlflow.sklearn.log_model(meta, name="stack_model")
            mlflow.lightgbm.log_model(lgb_pipeline.named_steps["model"], name="lgb_model")
            mlflow.xgboost.log_model(xgb_pipeline.named_steps["model"],  name="xgb_model")
            mlflow.log_artifact("models/feature_columns.pkl")
            mlflow.log_artifact("models/oof_predictions.csv")
            print("MLflow logged. Run ID:", mlflow.active_run().info.run_id)
    except Exception as e:
        print(f"MLflow skipped: {e}")

    # ---- Submission ----
    submission = pd.DataFrame({
        "SK_ID_CURR": test_ids.values,
        "TARGET":     meta.predict_proba(stack_test)[:, 1]
    })
    submission.to_csv("outputs/test_predictions.csv", index=False)
    print("Submission saved.")

# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    app_train, app_test = build_features()

    y      = app_train["TARGET"]
    X      = app_train.drop(columns=["TARGET", "SK_ID_CURR"])
    X_test = app_test.drop(columns=["SK_ID_CURR"])

    print(f"Train shape: {X.shape} | Test shape: {X_test.shape}")
    print(f"Target: {y.value_counts(normalize=True).round(3).to_dict()}")

    train_models(X, y, X_test, app_test["SK_ID_CURR"])


if __name__ == "__main__":
    main()