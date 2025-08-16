# app/streamlit_app.py
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="wide")

# ===== Paths =====
MODEL_FILE = Path("models/final_regularized_model.pkl")  # adjust if needed

# ===== Mappings: code -> human label (UI shows labels; model gets codes) =====
MAPS = {
    "checking_status": {
        "A11": "0 ≤ balance < 200 DM",
        "A12": "≥ 200 DM",
        "A13": "< 0 DM (overdrawn)",
        "A14": "No checking account",
    },
    "credit_history": {
        "A30": "No credits / all paid duly",
        "A31": "All credits at this bank paid duly",
        "A32": "Existing credits paid back duly till now",
        "A33": "Delay in paying off in the past",
        "A34": "Critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture / equipment",
        "A43": "Radio / TV",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100–<500 DM",
        "A63": "500–<1000 DM",
        "A64": "≥ 1000 DM",
        "A65": "Unknown / no savings",
    },
    "employment": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1–<4 years",
        "A74": "4–<7 years",
        "A75": "≥ 7 years",
    },
    "personal_status": {
        "A91": "Male: divorced/separated",
        "A92": "Male: single",
        "A93": "Male: married/widowed",
        "A94": "Female",
    },
    "other_parties": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property_magnitude": {
        "A121": "Real estate",
        "A122": "Life insurance",
        "A123": "Car",
        "A124": "Unknown / no property",
    },
    "other_payment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None",
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "For free",
    },
    "job": {
        "A171": "Unskilled (non-resident)",
        "A172": "Unskilled (resident)",
        "A173": "Skilled employee/official",
        "A174": "Management / self-employed",
    },
    "own_telephone": {
        "A191": "No",
        "A192": "Yes",
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No",
    },
}

# Expected input schema (order matters for safety, but ColumnTransformer also uses names)
CAT_COLS = list(MAPS.keys())
NUM_COLS = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]
ALL_COLUMNS = CAT_COLS + NUM_COLS

DEFAULTS_NUM = {
    "duration": 24,
    "credit_amount": 2500,
    "installment_commitment": 3,
    "residence_since": 2,
    "age": 35,
    "existing_credits": 1,
    "num_dependents": 1,
}

# ===== Helpers =====
def label_to_code(field: str, label: str) -> str:
    """Map a human label back to dataset code."""
    inv = {v: k for k, v in MAPS[field].items()}
    return inv[label]

def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing columns with sensible defaults and reorder to training schema."""
    df = df.copy()
    # fill categorical with first option
    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = next(iter(MAPS[c].keys()))
    # fill numerics with defaults
    for c, val in DEFAULTS_NUM.items():
        if c not in df.columns:
            df[c] = val
    # reorder
    return df[ALL_COLUMNS]

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        st.error(f"Model file not found at: {MODEL_FILE.resolve()}")
        st.stop()
    try:
        return joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()

# ===== UI =====
st.title("💳 Credit Risk Scoring Application")
st.caption("Estimate probability of default (PD) for loan applicants. "
           "Model trained on German Credit dataset. Default decision threshold = **0.35**.")

# Sidebar: threshold + template
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision Threshold (Approve if PD < threshold)", 0.0, 1.0, 0.35, 0.01)

    # Download a CSV template matching schema
    template = pd.DataFrame([{**{c: next(iter(MAPS[c].keys())) for c in CAT_COLS}, **DEFAULTS_NUM}])[ALL_COLUMNS]
    st.download_button(
        "Download CSV Template",
        data=template.to_csv(index=False),
        file_name="credit_scoring_template.csv",
        mime="text/csv",
    )

# Layout
tabs = st.tabs(["🧍 Single Applicant", "📦 Batch Scoring", "ℹ️ Model Info"])

# ===== Single Applicant =====
with tabs[0]:
    st.subheader("Applicant Information")

    # Numeric inputs
    n1, n2, n3, n4 = st.columns(4)
    with n1:
        duration = st.number_input("Duration of Credit (months)", 4, 72, DEFAULTS_NUM["duration"])
        credit_amount = st.number_input("Credit Amount", 250, 20000, DEFAULTS_NUM["credit_amount"])
    with n2:
        age = st.number_input("Age", 18, 85, DEFAULTS_NUM["age"])
        installment_commitment = st.number_input("Installment Commitment (%)", 1, 4, DEFAULTS_NUM["installment_commitment"])
    with n3:
        residence_since = st.number_input("Years at Residence", 1, 6, DEFAULTS_NUM["residence_since"])
        existing_credits = st.number_input("Existing Credits", 1, 4, DEFAULTS_NUM["existing_credits"])
    with n4:
        num_dependents = st.number_input("Number of Dependents", 0, 3, DEFAULTS_NUM["num_dependents"])

    st.markdown("**Categorical Inputs**")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        checking_status = label_to_code("checking_status", st.selectbox("Checking Status", list(MAPS["checking_status"].values())))
        credit_history  = label_to_code("credit_history",  st.selectbox("Credit History",  list(MAPS["credit_history"].values())))
        purpose         = label_to_code("purpose",         st.selectbox("Purpose",         list(MAPS["purpose"].values())))
        savings_status  = label_to_code("savings_status",  st.selectbox("Savings Status",  list(MAPS["savings_status"].values())))
    with c2:
        employment      = label_to_code("employment",      st.selectbox("Employment",      list(MAPS["employment"].values())))
        personal_status = label_to_code("personal_status", st.selectbox("Personal Status", list(MAPS["personal_status"].values())))
        other_parties   = label_to_code("other_parties",   st.selectbox("Other Parties",   list(MAPS["other_parties"].values())))
        property_mag    = label_to_code("property_magnitude", st.selectbox("Property Magnitude", list(MAPS["property_magnitude"].values())))
    with c3:
        other_pp        = label_to_code("other_payment_plans", st.selectbox("Other Payment Plans", list(MAPS["other_payment_plans"].values())))
        housing         = label_to_code("housing",         st.selectbox("Housing",         list(MAPS["housing"].values())))
        job             = label_to_code("job",             st.selectbox("Job",             list(MAPS["job"].values())))
    with c4:
        own_telephone   = label_to_code("own_telephone",   st.selectbox("Own Telephone",   list(MAPS["own_telephone"].values())))
        foreign_worker  = label_to_code("foreign_worker",  st.selectbox("Foreign Worker",  list(MAPS["foreign_worker"].values())))

    # Build row with codes
    row = pd.DataFrame([{
        "checking_status": checking_status,
        "credit_history": credit_history,
        "purpose": purpose,
        "savings_status": savings_status,
        "employment": employment,
        "personal_status": personal_status,
        "other_parties": other_parties,
        "property_magnitude": property_mag,
        "other_payment_plans": other_pp,
        "housing": housing,
        "job": job,
        "own_telephone": own_telephone,
        "foreign_worker": foreign_worker,
        "duration": duration,
        "credit_amount": credit_amount,
        "installment_commitment": installment_commitment,
        "residence_since": residence_since,
        "age": age,
        "existing_credits": existing_credits,
        "num_dependents": num_dependents,
    }])

    # Ensure schema & predict
    row = ensure_schema(row)

    if st.button("Predict Default Probability"):
        try:
            pd_bad = float(model.predict_proba(row)[:, 1][0])  # class 1 = bad
            decision = "✅ APPROVE" if pd_bad < threshold else "❌ REJECT"
            m1, m2, m3 = st.columns(3)
            m1.metric("PD (Bad=1)", f"{pd_bad:.3f}")
            m2.metric("Threshold", f"{threshold:.2f}")
            m3.metric("Decision", decision)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ===== Batch Scoring =====
with tabs[1]:
    st.subheader("Batch Scoring (Upload CSV)")
    st.caption("CSV columns can be in any order; missing columns will be auto-filled. "
               "Categorical values must be the dataset codes (e.g., A11, A12…).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up)
            df_in = ensure_schema(df_in)
            probs = model.predict_proba(df_in)[:, 1]
            out = df_in.copy()
            out["pd_bad"] = probs
            out["decision"] = np.where(out["pd_bad"] < threshold, "APPROVE", "REJECT")
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button("Download Scored CSV", out.to_csv(index=False), "scored_applicants.csv", "text/csv")
        except Exception as e:
            st.error(f"Batch scoring failed: {e}")

# ===== Model Info =====
with tabs[2]:
    st.subheader("Model Info")
    try:
        prep = model.named_steps.get("prep", None)
        clf  = model.named_steps.get("clf", None)
        st.write("Preprocessor:", type(prep).__name__ if prep is not None else "N/A")
        st.write("Classifier:", type(clf).__name__ if clf is not None else "N/A")
    except Exception:
        st.write("Pipeline details unavailable (non-standard object).")

    st.json({
        "threshold_default": 0.35,
        "expected_columns": ALL_COLUMNS,
        "note": "The model expects the original German Credit codes for categorical fields."
    })