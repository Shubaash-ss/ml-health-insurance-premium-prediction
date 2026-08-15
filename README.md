# Health Insurance Premium Predictor

A regression-based ML app that predicts health insurance premiums from applicant
demographics, lifestyle, and medical history — using two separate models split
by age group (≤25 and >25) for better accuracy across life stages.

## Live Demo
https://ml-health-insurance-premium-prediction-ss.streamlit.app

## About
Built as part of a structured machine learning course covering end-to-end
regression modeling and deployment. Extended with Streamlit Cloud deployment
and dependency fixes for Python 3.12 compatibility.

## What it does
- Collects 12 applicant inputs via a Streamlit form: age, dependants, income,
  genetic risk, insurance plan tier, employment status, gender, marital status,
  BMI category, smoking status, region, and medical history
- Converts medical history into a normalized risk score (weighted by condition
  severity — e.g. heart disease weighted higher than thyroid)
- One-hot encodes categorical fields and scales numeric fields using a
  pre-fitted `StandardScaler`
- Routes the prediction to one of two trained models depending on applicant
  age (≤25 vs >25), since risk patterns differ meaningfully across age groups
- Returns a predicted premium instantly in the UI

## Tech Stack
Python · scikit-learn · Streamlit · pandas · joblib

## Project Structure
```
├── artifacts/              # Trained models + scalers (young/rest split)
├── main.py                 # Streamlit UI
├── prediction_helper.py    # Preprocessing, risk scoring, prediction logic
├── requirements.txt
```

## Run locally
```
pip install -r requirements.txt
streamlit run main.py
```