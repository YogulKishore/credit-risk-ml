# Credit Risk Prediction System

A production-grade machine learning system for predicting credit default risk, built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset. Features a stacked ensemble of LightGBM, XGBoost, and CatBoost models served via a FastAPI backend and an interactive Gradio frontend.

🔗 **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/Betelgeuse4096/credit_risk)  
📦 **API:** [Railway](https://credit-risk-ml-production-8f03.up.railway.app/docs)  
📁 **Repo:** [GitHub](https://github.com/YogulKishore/credit-risk-ml)

---

## Architecture

```
Hugging Face Spaces          Railway (Docker)
────────────────────         ──────────────────────
Gradio Frontend     ──────▶  FastAPI Backend
                                  │
                                  ├── LightGBM
                                  ├── XGBoost
                                  ├── CatBoost
                                  └── Stacked Meta-Model
                                  │
                                  ▼
                             PostgreSQL Database
```

---

## Model Performance

All metrics evaluated on out-of-fold (OOF) predictions using 5-fold stratified cross-validation.

| Model       | ROC-AUC | PR-AUC | Log Loss |
|-------------|---------|--------|----------|
| LightGBM    | 0.7818  | 0.2734 | 0.2388   |
| XGBoost     | 0.7844  | 0.2765 | 0.2379   |
| CatBoost    | 0.7865  | 0.2794 | 0.2373   |
| **Stacked** | **0.7876** | **0.2820** | 0.2439 |

Optimal classification threshold tuned via F1 maximization on the PR curve:  
- **Best threshold:** 0.117  
- **Recall:** 0.437 &nbsp;|&nbsp; **Precision:** 0.276 &nbsp;|&nbsp; **F1:** 0.338

---

## Features

- **Predict** — fill in applicant details and get an instant default probability with per-model breakdown
- **Randomize** — generate synthetic applicants for quick testing
- **History** — view all past predictions stored in PostgreSQL
- **Dashboard** — risk distribution, score histograms, model comparisons, breakdowns by education, gender, and family status

---

## Feature Engineering

Features are engineered from 7 raw tables:

- `application_train/test` — core applicant and loan details
- `bureau` + `bureau_balance` — external credit history
- `previous_application` — past loan applications
- `POS_CASH_balance` — point-of-sale and cash loan history
- `installments_payments` — repayment history
- `credit_card_balance` — credit card usage

Key engineered features include credit/income ratio, annuity/income ratio, EXT_SOURCE mean, payment delay statistics, and bureau loan counts.

---

## SHAP Feature Importance

Top features driving predictions (LightGBM, SHAP TreeExplainer):

| Feature | Impact |
|---------|--------|
| EXT_SOURCE_MEAN | Strongest negative risk signal — higher scores = lower default |
| PAYMENT_DELAY_MAX | Late payments strongly increase risk |
| AMT_PAYMENT_SUM | Higher total payments = lower risk |
| CREDIT_ANNUITY_RATIO | Higher loan-to-annuity ratio increases risk |
| DAYS_EMPLOYED | Longer employment reduces risk |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | LightGBM, XGBoost, CatBoost, scikit-learn |
| Stacking | Logistic Regression meta-model |
| API | FastAPI, Uvicorn |
| Frontend | Gradio |
| Database | PostgreSQL (psycopg2) |
| Experiment Tracking | MLflow |
| Explainability | SHAP |
| Deployment | Docker, Railway, Hugging Face Spaces |

---

## Project Structure

```
credit-risk-ml/
├── api/
│   └── main.py              # FastAPI app — predict, history endpoints
├── src/
│   ├── features.py          # Feature engineering pipeline
│   ├── predict.py           # Single prediction logic
│   ├── db.py                # PostgreSQL schema and queries
│   ├── train.py             # Model training with 5-fold CV
│   ├── evaluate.py          # OOF evaluation and threshold tuning
│   ├── explain.py           # SHAP waterfall and summary plots
│   ├── log_mlflow.py        # MLflow experiment logging
│   └── bulk_generate.py     # Synthetic data generator
├── notebook/
│   └── eda_02.ipynb         # Exploratory data analysis
├── models/                  # Trained .pkl model files
├── Dockerfile               # Railway deployment
├── requirements.txt
└── README.md
```

---

## Running Locally

**1. Clone and install:**
```bash
git clone https://github.com/YogulKishore/credit-risk-ml.git
cd credit-risk-ml
pip install -r requirements.txt
```

**2. Add data** (download from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)) to `data/`

**3. Train models:**
```bash
python src/train.py
```

**4. Start API:**
```bash
uvicorn api.main:app --reload
```

**5. API docs:** `http://localhost:8000/docs`

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — Kaggle competition dataset.  
307,511 training samples | 121 raw features | ~8% positive class (default)
