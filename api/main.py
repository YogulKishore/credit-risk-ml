import sys
from pathlib import Path
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np

# FIX 1: point to project root (parent of api/) not api/ itself
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import predict_single, load_models, BEST_THRESHOLD
from src.db import (
    insert_application, fetch_all_applications,
    fetch_application, fetch_features, create_tables
)

# ------------------------------------------------
# App
# ------------------------------------------------

app = FastAPI(
    title="Credit Risk API",
    description="Ensemble ML model for credit default risk prediction",
    version="2.0.0"
)

# FIX 2: load models once at startup — not on every request
# stored in app state so all requests share the same loaded models
@app.on_event("startup")
def startup():
    print("Loading models...")
    app.state.models = load_models()
    print("Models loaded.")

    # FIX 7: wrap table creation so DB being down doesn't crash the app
    try:
        create_tables()
        print("DB tables ready.")
    except Exception as e:
        print(f"Warning: DB not available at startup — {e}")
        print("Predictions will still work but history won't be saved.")


# ------------------------------------------------
# Schemas
# ------------------------------------------------

class ApplicantInput(BaseModel):
    # Personal
    full_name:     str
    email:         Optional[str] = ""
    phone:         Optional[str] = ""
    address:       Optional[str] = ""
    date_of_birth: date

    # Loan details
    gender:           str
    income:           float = Field(gt=0, description="Annual income must be > 0")
    loan_amount:      float = Field(gt=0)
    annuity:          float = Field(gt=0)
    employment_years: int   = Field(ge=0)
    family_members:   int   = Field(gt=0)
    owns_car:         str
    owns_house:       str
    education:        str
    family_status:    str

    # External scores — must be between 0 and 1
    ext_source_1: float = Field(ge=0.0, le=1.0)
    ext_source_2: float = Field(ge=0.0, le=1.0)
    ext_source_3: float = Field(ge=0.0, le=1.0)

    # FIX 6: validate gender and owns fields
    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in ("M", "F", "XNA"):
            raise ValueError("gender must be M, F, or XNA")
        return v

    @field_validator("owns_car", "owns_house")
    @classmethod
    def validate_yn(cls, v):
        if v not in ("Y", "N"):
            raise ValueError("must be Y or N")
        return v


class PredictionResponse(BaseModel):
    applicant:      str
    stacked_score:  float
    lgb_score:      float
    xgb_score:      float
    cat_score:      float
    risk_label:     str
    is_default:     bool
    application_id: Optional[int] = None


# ------------------------------------------------
# Health check
# ------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "message": "Credit Risk API is running", "version": "2.0.0"}


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
        "CNT_FAM_MEMBERS":     [float(input.family_members)],
        "FLAG_OWN_CAR":        [input.owns_car],
        "FLAG_OWN_REALTY":     [input.owns_house],
        "NAME_EDUCATION_TYPE": [input.education],
        "NAME_FAMILY_STATUS":  [input.family_status],
        # FIX 3: don't use random values — let the imputer handle missing car age
        "OWN_CAR_AGE":         [None],
        "EXT_SOURCE_1":        [input.ext_source_1],
        "EXT_SOURCE_2":        [input.ext_source_2],
        "EXT_SOURCE_3":        [input.ext_source_3],
    })

    try:
        # FIX 2: pass pre-loaded models so no disk reads per request
        result = predict_single(app_df, models=app.state.models)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=str(e))

    # FIX 4: use risk_label from predict_single — don't recompute it
    label     = result["risk_label"]
    risk      = result["stacked"]
    is_default = result["is_default"]

    # FIX 5: engineered features for DB storage
    # guard against zero denominators
    engineered = {
        "credit_income_ratio":  round(input.loan_amount / input.income, 4),
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

    # save to DB — non-fatal if DB is down
    application_id = None
    try:
        application_id = insert_application(
            personal, inputs, engineered, result, result["feature_vector"]
        )
    except Exception as e:
        print(f"Warning: DB insert failed — {e}")

    return PredictionResponse(
        applicant=input.full_name,
        stacked_score=round(risk, 4),
        lgb_score=round(result["lgb"], 4),
        xgb_score=round(result["xgb"], 4),
        cat_score=round(result["cat"], 4),
        risk_label=label,
        is_default=bool(is_default),
        application_id=application_id,
    )


# ------------------------------------------------
# Get all applications
# ------------------------------------------------

@app.get("/applications")
def get_applications():
    try:
        df = fetch_all_applications()
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")


# ------------------------------------------------
# Get single application
# ------------------------------------------------

@app.get("/applications/{app_id}")
def get_application(app_id: int):
    try:
        row = fetch_application(app_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    return row


# ------------------------------------------------
# Get features for an application
# ------------------------------------------------

@app.get("/applications/{app_id}/features")
def get_application_features(app_id: int):
    try:
        row = fetch_features(app_id)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {e}")
    if not row:
        raise HTTPException(status_code=404, detail="Features not found")
    return row