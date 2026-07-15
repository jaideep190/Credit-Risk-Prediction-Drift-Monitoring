"""
Streamlit dashboard for the Credit Risk Prediction API.

Two tabs:
    1. Predict - a form that calls the live FastAPI /predict endpoint and
       shows the result with a risk gauge. This is the recruiter-facing
       demo: no code, no terminal, just fill in a form and click a button.
    2. Drift Monitoring - reads logged request data and will show PSI
       drift metrics over simulated time windows. Currently a placeholder
       until src/drift_monitor.py is built (next step) - showing fabricated
       drift numbers here would be dishonest, so it stays clearly labeled
       as not-yet-implemented rather than faked.

Run:
    streamlit run dashboard/app.py

Requires the FastAPI app to be running separately (default: http://localhost:8000).
"""

import json
import os

import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")
LOG_PATH = "logs/requests.jsonl"


def get_api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL)


def call_predict_api(payload: dict, api_url: str) -> dict:
    response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def render_risk_gauge(probability: float, threshold: float) -> go.Figure:
    color = "#C44E52" if probability >= threshold else "#55A868"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Predicted Default Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#E8F5E9"},
                    {"range": [threshold * 100, 100], "color": "#FFEBEE"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=300, margin=dict(t=50, b=10, l=20, r=20))
    return fig


def predict_tab():
    st.subheader("Applicant Risk Assessment")

    with st.sidebar:
        st.text_input("API URL", value=DEFAULT_API_URL, key="api_url")

    col1, col2 = st.columns(2)

    with col1:
        revolving_utilization = st.slider(
            "Revolving Utilization of Unsecured Lines", 0.0, 2.0, 0.3, 0.01,
            help="Total balance on credit cards/lines divided by credit limits"
        )
        age = st.number_input("Age", min_value=18, max_value=110, value=42)
        late_30_59 = st.number_input("Times 30-59 Days Past Due (last 2 years)", min_value=0, value=0)
        debt_ratio = st.slider("Debt Ratio", 0.0, 3.0, 0.35, 0.01)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5500, step=100)

    with col2:
        open_credit_lines = st.number_input("Number of Open Credit Lines and Loans", min_value=0, value=6)
        late_90 = st.number_input("Times 90+ Days Late", min_value=0, value=0)
        real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", min_value=0, value=1)
        late_60_89 = st.number_input("Times 60-89 Days Past Due (last 2 years)", min_value=0, value=0)
        dependents = st.number_input("Number of Dependents", min_value=0, value=2)

    if st.button("Predict Risk", type="primary"):
        payload = {
            "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "NumberOfDependents": dependents,
        }

        try:
            result = call_predict_api(payload, get_api_url())
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API at {get_api_url()}. Is it running? ({e})")
            return

        probability = result["default_probability"]
        threshold = result["decision_threshold"]
        is_high_risk = result["is_high_risk"]

        st.plotly_chart(render_risk_gauge(probability, threshold), use_container_width=True)

        if is_high_risk:
            st.error(f"HIGH RISK - flagged for review (probability {probability:.1%} >= threshold {threshold:.1%})")
        else:
            st.success(f"LOW RISK (probability {probability:.1%} < threshold {threshold:.1%})")

        st.caption(f"Model: {result['model_name']} | Decision threshold: {threshold:.4f}")


def drift_tab():
    st.subheader("Drift Monitoring")
    st.info(
        "This tab will show Population Stability Index (PSI) drift metrics comparing "
        "incoming request data against the training distribution, tracked over simulated "
        "time windows. Not yet implemented - see src/drift_monitor.py (coming next)."
    )

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            request_count = sum(1 for _ in f)
        st.metric("Logged prediction requests so far", request_count)
        st.caption(f"Reading from {LOG_PATH}")
    else:
        st.metric("Logged prediction requests so far", 0)
        st.caption("No requests logged yet - use the Predict tab or call the API to generate some.")


def main():
    st.title("Credit Risk Prediction Dashboard")
    tab1, tab2 = st.tabs(["Predict", "Drift Monitoring"])
    with tab1:
        predict_tab()
    with tab2:
        drift_tab()


if __name__ == "__main__":
    main()