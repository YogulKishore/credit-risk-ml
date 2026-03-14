# Credit Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-containerized-blue?logo=docker)
![LightGBM](https://img.shields.io/badge/LightGBM-ensemble-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-ensemble-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-ensemble-purple)

An end-to-end machine learning system for predicting credit default risk, built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle dataset. Features a stacked ensemble model served via a REST API with a full-stack web interface and persistent storage.

---


## Screenshots

### Application Form
![Form](screenshots/screenshot_form.png)

### Prediction Result
![Result](screenshots/screenshot_result.png)

### Submission History
![History](screenshots/screenshot_history.png)

---
## Architecture

```
┌─────────────────────┐        HTTP POST /predict        ┌─────────────────────────┐
│                     │ ──────────────────────────────►  │                         │
│  Streamlit Frontend │                                   │   FastAPI Backend       │
│  (app/streamlit_    │ ◄──────────────────────────────  │   (api/main.py)         │
│   app.py)           │        JSON response              │                         │
└─────────────────────┘                                   └───────────┬─────────────┘
                                                                      │
                                                          ┌───────────▼─────────────┐
                                                          │                         │
                                                          │   Ensemble ML Model     │
                                                          │   LGB + XGB + CatBoost  │
                                                          │   + Stacking Layer      │
                                                          │                         │
                                                          └───────────┬─────────────┘
                                                                      │
                                                          ┌───────────▼─────────────┐
                                                          │                         │
                                                          │   PostgreSQL (Docker)   │
                                                          │   - applications table  │
                                                          │   - application_features│
                                                          │     (167 features)      │
                                                          └─────────────────────────┘
```

---

## Model Performance

| Model | ROC-AUC | PR-AUC | Log Loss |
|---|---|---|---|
| LightGBM | 0.7818 | 0.2734 | 0.2388 |
| XGBoost | 0.7844 | 0.2765 | 0.2379 |
| CatBoost | 0.7863 | 0.2796 | 0.2373 |
| **Stacked Ensemble** | **0.7875** | **0.2820** | 0.2439 |

> Evaluated on out-of-fold (OOF) predictions using 5-fold stratified cross validation.
> Best F1 threshold: **0.119** — Precision: 0.279, Recall: 0.431, F1: 0.339

---

## Features

- **Stacked Ensemble** — LightGBM + XGBoost + CatBoost base models with Logistic Regression meta-model
- **167 engineered features** — application data, bureau history, POS cash, installments, credit card balances
- **FastAPI backend** — REST API with auto-generated Swagger docs at `/docs`
- **Streamlit frontend** — interactive form with randomize button, processing log, feature table
- **PostgreSQL storage** — all submissions + full 167-feature vectors stored per prediction
- **Docker** — Postgres runs in a container via `docker-compose`

---

## Project Structure

```
credit-risk-ml/
├── api/
│   └── main.py               # FastAPI app — /predict, /applications, /health
├── app/
│   └── streamlit_app.py      # Streamlit frontend
├── src/
│   ├── features.py           # Feature engineering pipeline
│   ├── predict.py            # Ensemble inference logic
│   ├── train.py              # Model training with 5-fold CV
│   ├── evaluate.py           # OOF evaluation + threshold tuning
│   ├── explain.py            # SHAP explainability
│   └── db.py                 # PostgreSQL connection + queries
├── models/
│   ├── lgb_model.pkl
│   ├── xgb_model.pkl
│   ├── cat_model.pkl
│   ├── stack_model.pkl
│   └── feature_columns.pkl
├── docker-compose.yml
└── requirements.txt
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Run prediction for an applicant |
| GET | `/applications` | Fetch all submissions |
| GET | `/applications/{id}` | Fetch single submission |
| GET | `/applications/{id}/features` | Fetch full 167-feature vector |

Interactive docs available at `http://localhost:8000/docs`

---

## Setup

### Prerequisites
- Python 3.10+
- Docker Desktop

### 1. Clone the repo
```bash
git clone https://github.com/YogulKishore/credit-risk-ml.git
cd credit-risk-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start PostgreSQL
```bash
docker-compose up -d
```

### 4. Initialise the database
```bash
python src/db.py
```

### 5. Download the dataset
Download from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place CSV files in `data/`

### 6. Train models *(optional — pretrained models included)*
```bash
python src/train.py
```

### 7. Start the API
```bash
uvicorn api.main:app --reload
```

### 8. Start the frontend
```bash
streamlit run app/streamlit_app.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | LightGBM, XGBoost, CatBoost, Scikit-learn |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Database | PostgreSQL 15 |
| Containerisation | Docker, Docker Compose |
| Data | Pandas, NumPy |
| Explainability | SHAP |

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — Kaggle competition dataset with 300,000+ loan applications across 9 related tables.
