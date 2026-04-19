import numpy as np
import pandas as pd


def preprocess_application(df):

    df = df.copy()

    drop_cols = ["APARTMENTS_MEDI", "APARTMENTS_MODE"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["FLAG_OWN_CAR"]    = df["FLAG_OWN_CAR"].map({"Y": 1, "N": 0})
    df["FLAG_OWN_REALTY"] = df["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0})

    df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].replace([64, 65], np.nan)

    df["AGE"] = df["DAYS_BIRTH"] / -365

    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"]      = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"]       / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"]      / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"]       / df["AMT_ANNUITY"].replace(0, np.nan)
    df["INCOME_PER_PERSON"]    = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)

    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    int_cols = df.select_dtypes("int64").columns
    df[int_cols] = df[int_cols].astype("int32")

    return df


def bureau_features(bureau):

    bureau = bureau.copy()

    bureau_counts = bureau.groupby("SK_ID_CURR").size().rename("BUREAU_LOAN_COUNT")

    bureau_agg = bureau.groupby("SK_ID_CURR").agg({
        "DAYS_CREDIT":            ["mean", "min", "max"],
        "AMT_CREDIT_SUM":         ["sum", "mean"],
        "AMT_CREDIT_SUM_DEBT":    ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum"],
    })

    bureau_agg.columns = ["_".join(col).upper() for col in bureau_agg.columns]
    bureau_agg["BUREAU_LOAN_COUNT"] = bureau_counts

    active_loans = (
        bureau[bureau["CREDIT_ACTIVE"] == "Active"]
        .groupby("SK_ID_CURR").size().rename("ACTIVE_LOAN_COUNT")
    )

    bureau_agg = bureau_agg.merge(active_loans, left_index=True, right_index=True, how="left")
    bureau_agg["ACTIVE_LOAN_COUNT"] = bureau_agg["ACTIVE_LOAN_COUNT"].fillna(0)

    return bureau_agg.reset_index()


def bureau_balance_features(bureau, bureau_balance):

    bureau        = bureau.copy()
    bureau_balance = bureau_balance.copy()

    bureau_balance["STATUS_NUM"] = bureau_balance["STATUS"].replace({
        "X": np.nan, "C": 0, "0": 0,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5
    })

    bureau_bal_agg = bureau_balance.groupby("SK_ID_BUREAU").agg({
        "MONTHS_BALANCE": ["min", "max"],
        "STATUS_NUM":     ["mean", "max"]
    })

    bureau_bal_agg.columns = ["_".join(col).upper() for col in bureau_bal_agg.columns]

    bureau = bureau.merge(bureau_bal_agg, how="left", on="SK_ID_BUREAU")

    bureau_feat = bureau.groupby("SK_ID_CURR").agg({
        "STATUS_NUM_MAX":     ["max", "mean"],
        "MONTHS_BALANCE_MIN": ["min"]
    })

    bureau_feat.columns = ["_".join(col).upper() for col in bureau_feat.columns]

    return bureau_feat.reset_index()


def previous_application_features(prev):

    prev = prev.copy()

    prev_count = prev.groupby("SK_ID_CURR").size().rename("PREV_APP_COUNT")

    approved = (
        prev[prev["NAME_CONTRACT_STATUS"] == "Approved"]
        .groupby("SK_ID_CURR").size().rename("APPROVED_COUNT")
    )

    refused = (
        prev[prev["NAME_CONTRACT_STATUS"] == "Refused"]
        .groupby("SK_ID_CURR").size().rename("REFUSED_COUNT")
    )

    prev_agg = prev.groupby("SK_ID_CURR").agg({
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT":      ["mean", "max"],
        "AMT_ANNUITY":     ["mean"],
        "DAYS_DECISION":   ["mean", "min"]
    })

    prev_agg.columns = ["_".join(col).upper() for col in prev_agg.columns]

    prev_agg = prev_agg.merge(prev_count, left_index=True, right_index=True, how="left")
    prev_agg = prev_agg.merge(approved,   left_index=True, right_index=True, how="left")
    prev_agg = prev_agg.merge(refused,    left_index=True, right_index=True, how="left")

    prev_agg["PREV_APP_COUNT"]  = prev_agg["PREV_APP_COUNT"].fillna(0)
    prev_agg["APPROVED_COUNT"]  = prev_agg["APPROVED_COUNT"].fillna(0)
    prev_agg["REFUSED_COUNT"]   = prev_agg["REFUSED_COUNT"].fillna(0)

    return prev_agg.reset_index()


def pos_features(pos):

    pos = pos.copy()

    pos_agg = pos.groupby("SK_ID_CURR").agg({
        "MONTHS_BALANCE":        ["min", "max"],
        "CNT_INSTALMENT":        ["mean", "max"],
        "CNT_INSTALMENT_FUTURE": ["mean"],
        "SK_DPD":                ["max", "mean"],
        "SK_DPD_DEF":            ["max"],
    })

    pos_agg.columns = ["_".join(col).upper() for col in pos_agg.columns]

    return pos_agg.reset_index()


def installment_features(installments):

    installments = installments.copy()

    installments["PAYMENT_DELAY"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    )

    inst_agg = installments.groupby("SK_ID_CURR").agg({
        "PAYMENT_DELAY":  ["mean", "max"],
        "AMT_PAYMENT":    ["sum", "mean"],
        "AMT_INSTALMENT": ["sum", "mean"],
    })

    inst_agg.columns = ["_".join(col).upper() for col in inst_agg.columns]

    return inst_agg.reset_index()


def credit_card_features(cc):

    cc = cc.copy()

    cc_agg = cc.groupby("SK_ID_CURR").agg({
        "AMT_BALANCE":             ["mean", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean"],
        "SK_DPD":                  ["max", "mean"],
    })

    cc_agg.columns = ["_".join(col).upper() for col in cc_agg.columns]

    return cc_agg.reset_index()