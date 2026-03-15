import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from train import load_data
from features import (
    preprocess_application,
    bureau_features,
    bureau_balance_features,
    previous_application_features,
    pos_features,
    installment_features,
    credit_card_features,
)


def explain():

    print("Loading raw data...")

    (
        application_train,
        application_test,
        bureau,
        bureau_balance,
        previous_application,
        pos_cash,
        installments,
        credit_card,
    ) = load_data()

    print("Building feature table...")

    # application features
    app = preprocess_application(application_train)

    # bureau features
    bureau_agg = bureau_features(bureau)

    # bureau balance features
    bureau_bal_agg = bureau_balance_features(bureau, bureau_balance)

    # previous applications
    prev_agg = previous_application_features(previous_application)

    # POS
    pos_agg = pos_features(pos_cash)

    # installments
    inst_agg = installment_features(installments)

    # credit card
    cc_agg = credit_card_features(credit_card)

    # merge all engineered features
    df = app.merge(bureau_agg, on="SK_ID_CURR", how="left")
    df = df.merge(bureau_bal_agg, on="SK_ID_CURR", how="left")
    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
    df = df.merge(pos_agg, on="SK_ID_CURR", how="left")
    df = df.merge(inst_agg, on="SK_ID_CURR", how="left")
    df = df.merge(cc_agg, on="SK_ID_CURR", how="left")

    # split target
    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    print("Loading trained pipeline...")

    lgb_pipeline = joblib.load("models/lgb_model.pkl")

    preprocess = lgb_pipeline.named_steps["preprocess"]
    model = lgb_pipeline.named_steps["model"]

    print("Transforming features using pipeline...")

    X_transformed = preprocess.transform(X)

    feature_names = preprocess.get_feature_names_out()

    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

    print("Creating SHAP explainer...")

    explainer = shap.TreeExplainer(model)

    # use sample for faster SHAP
    sample = X_transformed.sample(1000, random_state=42)

    shap_values = explainer(sample)

    print("Generating SHAP waterfall plot...")

    plt.figure(figsize=(12,6))
    shap.plots.waterfall(shap_values[0], max_display=10)

    print("Generating SHAP summary plot...")

    plt.figure(figsize=(12,8))
    shap.summary_plot(shap_values, sample, max_display=20)


if __name__ == "__main__":
    explain()