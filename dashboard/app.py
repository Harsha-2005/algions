"""
Student At-Risk Prediction — Streamlit dashboard.

This is a thin UI layer only. It holds no model-loading or scoring
logic of its own — every prediction is a real HTTP call to the FastAPI
backend (api/main.py), so there is exactly one place the model is
loaded and exactly one place the SHAP explanation is computed. This
dashboard exists to make that same real prediction visible and
explorable for a human reviewer, not to duplicate it.

Run (two terminals):
    python -m uvicorn api.main:app --port 8000
    python -m streamlit run dashboard/app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Three real students from the test set, used throughout the technical
# report (reports/technical_report.md, Section 8) — lets a reviewer
# reproduce the exact findings discussed there with one click.
PRESETS = {
    "False negative (BBB — perfect score, then collapsed)": {
        "code_module": "BBB", "total_clicks": 424.0, "distinct_active_days": 23.0,
        "clicks_last_7d": 10.0, "clicks_first_7d": 243.0,
        "num_assessments_due": 1.0, "num_submitted": 1.0, "num_missed": 0.0,
        "num_late": 0.0, "has_graded_work": True, "mean_score_so_far": 100.0,
        "date_registration": -45.0, "num_of_prev_attempts": 0.0, "studied_credits": 60.0,
    },
    "Obvious at-risk (AAA — missed assessment, barely engaged)": {
        "code_module": "AAA", "total_clicks": 3.0, "distinct_active_days": 1.0,
        "clicks_last_7d": 3.0, "clicks_first_7d": 0.0,
        "num_assessments_due": 1.0, "num_submitted": 0.0, "num_missed": 1.0,
        "num_late": 0.0, "has_graded_work": False, "mean_score_so_far": 0.0,
        "date_registration": -10.0, "num_of_prev_attempts": 1.0, "studied_credits": 120.0,
    },
    "Thriving (AAA — strong score, heavy engagement)": {
        "code_module": "AAA", "total_clicks": 774.0, "distinct_active_days": 28.0,
        "clicks_last_7d": 143.0, "clicks_first_7d": 416.0,
        "num_assessments_due": 1.0, "num_submitted": 1.0, "num_missed": 0.0,
        "num_late": 0.0, "has_graded_work": True, "mean_score_so_far": 87.0,
        "date_registration": -38.0, "num_of_prev_attempts": 0.0, "studied_credits": 60.0,
    },
}
FIELD_DEFAULTS = next(iter(PRESETS.values()))  # sensible first-load defaults

st.set_page_config(page_title="Student At-Risk Dashboard", page_icon="🎓", layout="centered")
st.title("🎓 Student At-Risk Prediction")
st.caption(
    "Decision-support tool for School Management. Every score comes from a "
    "real call to the inference API below — nothing on this page is "
    "hard-coded."
)

# --- API health check ---
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    if health.get("model_loaded"):
        st.success(f"Connected to inference API at {API_URL} — model loaded.")
    else:
        st.warning(f"API at {API_URL} is reachable but reports no model loaded.")
except requests.exceptions.RequestException:
    st.error(
        f"Cannot reach the inference API at {API_URL}. Start it first with:\n\n"
        f"`python -m uvicorn api.main:app --port 8000`"
    )
    st.stop()

# --- Preset loader (for the three cases discussed in the technical report) ---
st.subheader("Try a real student from the test set")
preset_cols = st.columns(len(PRESETS))
for col, (label, values) in zip(preset_cols, PRESETS.items()):
    if col.button(label.split(" (")[0], help=label, use_container_width=True):
        for k, v in values.items():
            st.session_state[k] = v

st.divider()
st.subheader("Or enter a student's profile manually")

for k, v in FIELD_DEFAULTS.items():
    st.session_state.setdefault(k, v)

col1, col2 = st.columns(2)
with col1:
    st.selectbox("Course module", ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"], key="code_module")
    st.number_input("Total VLE clicks by cutoff", min_value=0.0, key="total_clicks")
    st.number_input("Distinct active days", min_value=0.0, key="distinct_active_days")
    st.number_input("Clicks in first 7 days", min_value=0.0, key="clicks_first_7d")
    st.number_input("Clicks in last 7 days", min_value=0.0, key="clicks_last_7d")
    st.caption(f"Engagement trend (auto): {st.session_state['clicks_last_7d'] - st.session_state['clicks_first_7d']:+.0f}")
with col2:
    st.number_input("Assessments due by cutoff", min_value=0.0, key="num_assessments_due")
    st.number_input("Assessments submitted", min_value=0.0, key="num_submitted")
    st.number_input("Assessments missed", min_value=0.0, key="num_missed")
    st.number_input("Late submissions", min_value=0.0, key="num_late")
    st.checkbox("Has graded work by cutoff?", key="has_graded_work")
    if st.session_state["has_graded_work"]:
        st.number_input("Mean score so far", min_value=0.0, max_value=100.0, key="mean_score_so_far")

st.number_input("Days registered before course start (negative = early)", key="date_registration")
st.number_input("Previous attempts", min_value=0.0, key="num_of_prev_attempts")
st.number_input("Studied credits", min_value=0.0, key="studied_credits")

if st.button("Predict risk", type="primary"):
    num_due = st.session_state["num_assessments_due"]
    num_sub = st.session_state["num_submitted"]
    payload = {
        "code_module": st.session_state["code_module"],
        "total_clicks": st.session_state["total_clicks"],
        "distinct_active_days": st.session_state["distinct_active_days"],
        "clicks_last_7d": st.session_state["clicks_last_7d"],
        "clicks_first_7d": st.session_state["clicks_first_7d"],
        "engagement_trend": st.session_state["clicks_last_7d"] - st.session_state["clicks_first_7d"],
        "num_assessments_due": num_due,
        "num_submitted": num_sub,
        "num_missed": st.session_state["num_missed"],
        "num_late": st.session_state["num_late"],
        "mean_score_so_far": st.session_state["mean_score_so_far"] if st.session_state["has_graded_work"] else None,
        "submission_rate": (num_sub / num_due) if num_due else 0.0,
        "date_registration": st.session_state["date_registration"],
        "num_of_prev_attempts": st.session_state["num_of_prev_attempts"],
        "studied_credits": st.session_state["studied_credits"],
    }

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction request failed: {e}")
        st.stop()

    st.divider()
    band = result["risk_band"]
    prob = result["probability_at_risk"]
    band_color = {"Low risk": "green", "Moderate risk": "orange", "High risk": "red"}[band]

    st.markdown(f"## Risk band: :{band_color}[{band}]")
    st.progress(min(prob, 1.0), text=f"Probability at risk: {prob:.1%}")

    st.subheader("Most influential signals (live SHAP explanation)")
    sig_df = pd.DataFrame(result["top_influential_signals"]).set_index("feature")
    st.bar_chart(sig_df["shap_value"])

    st.info(f"⚠️ {result['human_review_note']}")
