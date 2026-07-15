"""
FastAPI application serving the credit risk model.

Loads, once at startup:
    - the champion model (LightGBM_baseline)
    - preprocessing_artifacts.json (medians, caps - fitted on train only)
    - decision_threshold.json (selected threshold + which model to use)
    - feature_columns.json (exact column order the model expects)

Exposes:
    POST /predict - returns a default probability + high-risk flag for one applicant
    GET  /health   - basic liveness check

Every request is also logged (raw feature values only, no prediction) to
logs/requests.jsonl for the drift monitoring module to consume later.
"""

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.preprocessing import RAW_FEATURE_COLUMNS, apply_preprocessing

from app.schemas import ApplicantFeatures, ExplanationResponse, PredictionResponse

MODELS_DIR = "models"
LOGS_DIR = "logs"
REQUEST_LOG_PATH = os.path.join(LOGS_DIR, "requests.jsonl")

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts probability of serious delinquency within 2 years, with request logging for drift monitoring.",
    version="1.0.0",
)

_state = {}


@app.on_event("startup")
def load_artifacts():
    with open(os.path.join(MODELS_DIR, "decision_threshold.json")) as f:
        decision_config = json.load(f)

    model_name = decision_config["champion_model"]
    model = joblib.load(os.path.join(MODELS_DIR, f"{model_name}.pkl"))

    with open(os.path.join(MODELS_DIR, "preprocessing_artifacts.json")) as f:
        preprocessing_artifacts = json.load(f)

    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    os.makedirs(LOGS_DIR, exist_ok=True)

    _state["model"] = model
    _state["model_name"] = model_name
    _state["threshold"] = decision_config["selected_threshold"]
    _state["preprocessing_artifacts"] = preprocessing_artifacts
    _state["feature_columns"] = feature_columns

    # Tree-path-dependent explainer: exact for tree ensembles, no background
    # dataset needed, fast enough to run per-request. Values are in the
    # model's raw margin (log-odds) space, not probability space directly -
    # this is the standard/default behavior for SHAP on tree classifiers.
    _state["shap_explainer"] = shap.TreeExplainer(model)

    print(f"Loaded model: {model_name}, threshold: {_state['threshold']}")


def log_request(raw_values: dict, window: str = None) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window": window,
        **raw_values,
    }
    with open(REQUEST_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantFeatures, window: str = None):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_dict = applicant.dict(by_alias=True)
    df = pd.DataFrame([raw_dict])[RAW_FEATURE_COLUMNS]

    processed_df = apply_preprocessing(df, _state["preprocessing_artifacts"])
    processed_df = processed_df[_state["feature_columns"]]

    probability = float(_state["model"].predict_proba(processed_df)[0, 1])
    is_high_risk = probability >= _state["threshold"]

    log_request(raw_dict, window=window)

    return PredictionResponse(
        default_probability=round(probability, 4),
        is_high_risk=is_high_risk,
        decision_threshold=_state["threshold"],
        model_name=_state["model_name"],
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(applicant: ApplicantFeatures, window: str = None):
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_dict = applicant.dict(by_alias=True)
    df = pd.DataFrame([raw_dict])[RAW_FEATURE_COLUMNS]

    processed_df = apply_preprocessing(df, _state["preprocessing_artifacts"])
    processed_df = processed_df[_state["feature_columns"]]

    probability = float(_state["model"].predict_proba(processed_df)[0, 1])
    is_high_risk = probability >= _state["threshold"]

    log_request(raw_dict, window=window)

    explainer = _state["shap_explainer"]
    raw_shap_values = explainer.shap_values(processed_df)
    raw_expected_value = explainer.expected_value

    # shap's return shape varies by version: some return a list [class0, class1]
    # for binary classifiers, some return a single array for the positive class.
    # Handle both so this doesn't silently break on a different shap version.
    if isinstance(raw_shap_values, list):
        instance_shap_values = raw_shap_values[1][0]
        base_value = float(raw_expected_value[1])
    else:
        instance_shap_values = raw_shap_values[0]
        base_value = float(raw_expected_value)

    shap_contributions = {
        col: float(val) for col, val in zip(_state["feature_columns"], instance_shap_values)
    }

    return ExplanationResponse(
        default_probability=round(probability, 4),
        is_high_risk=is_high_risk,
        decision_threshold=_state["threshold"],
        model_name=_state["model_name"],
        base_value=round(base_value, 4),
        shap_contributions=shap_contributions,
    )