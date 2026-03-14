import sys
from pathlib import Path
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.predict import predict_single
from src.db import insert_application, fetch_all_applications, fetch_application, fetch_features, create_tables

# ------------------------------------------------
# App
# ------------------------------------------------

app = FastAPI(
    title="Credit Risk API",
    description="Ensemble ML model for credit default risk prediction",
    version="1.0.0"
)

create_tables()


# ------------------------------------------------
# Schemas
# ------------------------------------------------

class ApplicantInput(BaseModel):
    # Personal
    full_name: str
    email: Optional[str] = ""
    phone: Optional[str] = ""
    address: Optional[str] = ""
    date_of_birth: date

    # Loan details
    gender: str
    income: float
    loan_amount: float
    annuity: float
    employment_years: int
    family_members: int
    owns_car: str
    owns_house: str
    education: str
    family_status: str

    # External scores
    ext_source_1: float
    ext_source_2: float
    ext_source_3: float


class PredictionResponse(BaseModel):
    applicant: str
    stacked_score: float
    lgb_score: float
    xgb_score: float
    cat_score: float
    risk_label: str
    application_id: int


# ------------------------------------------------
# Health check
# ------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "message": "Credit Risk API is running"}


# ------------------------------------------------
# Predict
# ------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(input: ApplicantInput):

    age = (date.today() - input.date_of_birth).days // 365

    app_df = pd.DataFrame({
        "SK_ID_CURR":          [999999],
        "CODE_GENDER":         [input.gender],
        "DAYS_BIRTH":          [-age * 365],
        "DAYS_EMPLOYED":       [-input.employment_years * 365],
        "AMT_INCOME_TOTAL":    [input.income],
        "AMT_CREDIT":          [input.loan_amount],
        "AMT_ANNUITY":         [input.annuity],
        "CNT_FAM_MEMBERS":     [input.family_members],
        "FLAG_OWN_CAR":        [input.owns_car],
        "FLAG_OWN_REALTY":     [input.owns_house],
        "NAME_EDUCATION_TYPE": [input.education],
        "NAME_FAMILY_STATUS":  [input.family_status],
        "OWN_CAR_AGE":         [np.random.randint(0, 20) if input.owns_car == "Y" else None],
        "EXT_SOURCE_1":        [input.ext_source_1],
        "EXT_SOURCE_2":        [input.ext_source_2],
        "EXT_SOURCE_3":        [input.ext_source_3],
    })

    try:
        result = predict_single(app_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    risk = result["stacked"]
    if risk < 0.2:
        label = "Low Risk"
    elif risk < 0.5:
        label = "Medium Risk"
    else:
        label = "High Risk"

    engineered = {
        "credit_income_ratio": round(input.loan_amount / input.income, 4),
        "annuity_income_ratio": round(input.annuity / input.income, 4),
        "credit_annuity_ratio": round(input.loan_amount / input.annuity, 4),
        "income_per_person":    round(input.income / input.family_members, 2),
        "ext_source_mean":      round((input.ext_source_1 + input.ext_source_2 + input.ext_source_3) / 3, 4),
    }

    personal = {
        "full_name":     input.full_name,
        "email":         input.email,
        "phone":         input.phone,
        "address":       input.address,
        "date_of_birth": input.date_of_birth,
    }

    inputs = {
        "gender":           input.gender,
        "age":              age,
        "income":           input.income,
        "loan_amount":      input.loan_amount,
        "annuity":          input.annuity,
        "employment_years": input.employment_years,
        "family_members":   input.family_members,
        "owns_car":         input.owns_car,
        "owns_house":       input.owns_house,
        "education":        input.education,
        "family_status":    input.family_status,
        "ext_source_1":     input.ext_source_1,
        "ext_source_2":     input.ext_source_2,
        "ext_source_3":     input.ext_source_3,
    }

    application_id = insert_application(personal, inputs, engineered, result, result["feature_vector"])

    return PredictionResponse(
        applicant=input.full_name,
        stacked_score=round(risk, 4),
        lgb_score=round(result["lgb"], 4),
        xgb_score=round(result["xgb"], 4),
        cat_score=round(result["cat"], 4),
        risk_label=label,
        application_id=application_id,
    )


# ------------------------------------------------
# Get all applications
# ------------------------------------------------

@app.get("/applications")
def get_applications():
    df = fetch_all_applications()
    return df.to_dict(orient="records")


# ------------------------------------------------
# Get single application
# ------------------------------------------------

@app.get("/applications/{app_id}")
def get_application(app_id: int):
    row = fetch_application(app_id)
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    return row


# ------------------------------------------------
# Get features for an application
# ------------------------------------------------

@app.get("/applications/{app_id}/features")
def get_application_features(app_id: int):
    row = fetch_features(app_id)
    if not row:
        raise HTTPException(status_code=404, detail="Features not found")
    return row