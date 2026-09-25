# Dynamic DS — Explainable ML Pipeline & Baseline Engine

**Dynamic DS** is a production-shaped, explainable machine learning platform designed to provide rigorous baseline testing, automated exploratory data analysis (EDA), and data-quality profiling. Unlike "black-box" AutoML tools, Dynamic DS enforces transparency at every step: all data transformations, feature engineering choices, and model evaluations are fully visible and justifiable. 

This platform prevents data leakage through strict cross-validation boundaries and chronological splits, ensuring that every model's performance is strictly compared against naive baselines to rigorously validate predictive power.
---

## 1. The Problem We Solve
Most "upload a CSV, get a model" tools obscure the intermediate steps. Dynamic DS keeps every decision visible: 
- What semantic type each column was inferred as and why.
- What data-quality issues exist and the recommended actions.
- How the baseline ML model compares to doing absolutely nothing (Dummy baseline) so claims of "the model works" are backed by empirical evidence.

## 2. System Architecture

```text
frontend (React 19 + TypeScript, built with Vite, served by FastAPI)
        │  REST + JSON
        ▼
FastAPI (backend/app/main.py)
        │
        ├─ datasets API  → dataset processing → EDA / data-quality engine
        │
        └─ experiments API → task selection → ML pipeline → evaluation
                                                     │
                                                     ├─ model registry (JSON + joblib)
                                                     └─ MLflow (optional mirror)
```

## 3. Core Features

- **Automated Profiling & Data Quality**: Detects missingness bands, duplicate rows, constant/near-constant columns, identifier tiers, placeholder normalization, and formatting anomalies (currency, email, URL).
- **Comprehensive EDA**: Computes numerical stats, KDEs, categorical cardinality, Cramér's V, and Pearson/Spearman correlations. Heuristically detects target leakage.
- **Standardized Warnings Engine**: Every dataset row receives a severity/issue/recommendation mapping for immediate actionability.
- **Rigorous Modeling**: Transitions from Dummy → Logistic/Linear Regression → Random Forest → XGBoost. Utilizes k-fold CV on train + untouched hold-out test splits. Binary and multiclass classification are natively supported.
- **Time-Series Forecasting**: Naive + seasonal-naive baselines, exponential smoothing, and lag-feature XGBoost. Strictly chronological train/validation/test splits. No random shuffling.
- **Explainability**: SHAP for tree models, linear coefficients, permutation importance, and local occlusion-based explanations for single predictions.
- **Experiment Tracking**: Training runs are versioned in a Model Registry (Development / Candidate / Production / Archived).
- **Automated Reporting**: Generates self-contained HTML reports for EDA and model evaluation.

## 4. ML Methodology & Leakage Prevention

- **Baselines are Mandatory**: DummyClassifier/Regressor for standard ML; naive and seasonal-naive for forecasting. Leaderboards explicitly state if the leading model statistically beats the baseline.
- **Strictly Leak-Free Pipelines**: All imputers, scalers, and encoders are fitted **inside** the CV fold and on the training split only.
- **Chronological Splitting**: Forecasting uses a strict chronological 70/15/15 split. The lag-feature model is rigorously tested to ensure identical forecasts regardless of "future" actual values.
- **Conservative Feature Engineering**: Log1p transformations for right-skewed numeric columns, one-hot encoding capped at 30 categories, and cyclical calendar features (sin/cos) for datetime columns.

## 5. Tech Stack

- **Frontend**: React 19, TypeScript, Vite, React Router, Recharts, Modern CSS (Glassmorphism, Inter/Outfit typography).
- **Backend**: Python 3.11+, FastAPI, pandas, NumPy, scikit-learn, XGBoost, statsmodels, SHAP.
- **Storage/Tracking**: Flat files (JSON + joblib), MLflow integration ready, Postgres-provisioned for future migration.

## 6. Installation & Setup

**Local Development:**
```bash
# 1. Build the frontend
cd frontend
npm install
npm run build

# 2. Run the backend
cd ../backend
python -m venv venv
source venv/bin/activate  # (or venv\Scripts\Activate on Windows)
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Open `http://localhost:8000/`.

## 7. Testing

The backend is fortified by a comprehensive test suite covering type inference, missing-value calculations, leak-free preprocessing, chronological splitting, and the full API surface.

```bash
cd backend
pytest -q
```
*Currently 49/49 tests passing end-to-end.*
