# app.py
# CardioSearch AI
# Streamlit Front-End for A* Heart Disease Diagnosis

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from astar_engine import astar_search

st.set_page_config(page_title="CardioSearch AI", page_icon="❤️")

@st.cache(allow_output_mutation=True)
def load_model():
    df = pd.read_csv("heart.csv")

    df = df[df["ca"] < 4]
    df = df[df["thal"] > 0]

    X = df.drop("target", axis=1)
    y = df["target"]

    model = RandomForestClassifier(
        max_depth=5,
        min_samples_leaf=4,
        random_state=42
    )
    model.fit(X, y)

    return model, X.columns.tolist()



model, feature_names = load_model()

st.title("❤️ Heart Disease Risk Predictor")
st.subheader("A* Search-Based Heart Disease Diagnosis System")



# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("🧠 AI System Info")

    st.write("**Algorithm:** A* Search")
    st.write("**Heuristic:** Random Forest Risk %")
    st.write("**Dataset:** UCI Heart Disease")

    st.write("---")

    st.write("### Features Used")
    st.write("✔ Chest Pain Type")
    st.write("✔ Blocked Vessels")
    st.write("✔ ECG Slope")
    st.write("✔ Cholesterol")
    st.write("✔ Age")
    st.write("✔ Exercise Angina")


# -----------------------------
# Upload File
# -----------------------------
st.subheader("📂 Upload Patient CSV")

uploaded_file = st.file_uploader(
    "Upload one patient record (.csv)",
    type=["csv"]
)


# -----------------------------
# Run Diagnosis
# -----------------------------
if uploaded_file:

    try:
        patient_data = pd.read_csv(uploaded_file)

        # Validate one row
        if len(patient_data) != 1:
            st.error("Please upload ONLY one patient row.")
            st.stop()

        # Validate columns
        if list(patient_data.columns) != feature_names:
            st.error("CSV columns do not match dataset format.")
            st.write("Expected:")
            st.write(feature_names)
            st.stop()

        # Run Search
        with st.spinner("Running A* Search Diagnosis..."):
            result = astar_search(patient_data, model)

        # -----------------------------
        # Search Trace
        # -----------------------------
        st.subheader("🔍 Search Trace")

        for step in result["path"]:
            st.write(step)

        # -----------------------------
        # Metrics
        # -----------------------------
        st.subheader("📊 Search Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Path Cost g(n)", result["cost"])
        col2.metric("Heuristic h(n)", result["heuristic"])
        col3.metric("Total Score f(n)", result["score"])

        # -----------------------------
        # Final Risk %
        # -----------------------------
        risk = round(model.predict_proba(patient_data)[0][1] * 100, 1)

        st.metric("Risk Confidence", f"{risk}%")

        if not result["goal_reached"]:
            st.info(
                "A* did not reach a high-risk goal state, so the app is "
                "showing the strongest findings discovered for this patient."
            )

        # -----------------------------
        # Final Diagnosis
        # -----------------------------
        st.subheader("🩺 Final Diagnosis")

        if risk >= 80:
            st.error("⚠️ HIGH RISK HEART DISEASE")
            st.warning(
                "Immediate cardiology consultation recommended."
            )

        elif risk >= 50:
            st.warning("⚠️ MODERATE RISK")
            st.info(
                "Further testing recommended."
            )

        else:
            st.success("✅ LOW RISK")
            st.info(
                "Maintain healthy lifestyle and routine checkups."
            )

        # -----------------------------
        # Rules Triggered
        # -----------------------------
        st.subheader("📌 Clinical Findings")

        if result["rules"]:
            for rule in result["rules"]:
                st.write("•", rule.replace("_", " ").title())
        else:
            st.write("No rule-based findings were triggered.")

    except Exception as e:
        st.error(f"Error: {e}")
