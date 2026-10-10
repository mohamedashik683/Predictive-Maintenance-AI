
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "predictive_maintenance_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #111c30);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background-color: #101a2b;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}
[data-testid="stMetric"] {
    background: #17263b;
    border: 1px solid #2b405d;
    padding: 18px;
    border-radius: 14px;
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #cbd5e1 !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
}
.stApp h1, .stApp h2, .stApp h3,
.stApp p, .stApp label, .stApp span {
    color: #e2e8f0;
}
.stButton > button {
    background: #1687f8;
    color: white !important;
    border: 0;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    saved_data = joblib.load(MODEL_PATH)
    if isinstance(saved_data, dict):
        return saved_data["model"]
    return saved_data


st.title("⚙️ Predictive Maintenance AI")
st.caption("SMART INDUSTRIAL MONITORING • MACHINE LEARNING")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run train_model.py first.")
    st.stop()

model = load_model()

# Sidebar inputs
st.sidebar.title("🛠️ Machine Configuration")
st.sidebar.caption("Enter the machine sensor readings")

machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.sidebar.number_input(
    "Air Temperature (K)", 290.0, 310.0, 300.0, step=0.1
)
process_temp = st.sidebar.number_input(
    "Process Temperature (K)", 300.0, 320.0, 310.0, step=0.1
)
rotational_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)", 1000, 3000, 1500, step=10
)
torque = st.sidebar.number_input(
    "Torque (Nm)", 3.0, 80.0, 40.0, step=0.5
)
tool_wear = st.sidebar.number_input(
    "Tool Wear (minutes)", 0, 300, 100, step=1
)

input_data = pd.DataFrame([{
    "Type": machine_type,
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear
}])

st.markdown("## 📊 Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Machine Type", machine_type)
c2.metric("Air Temperature", f"{air_temp:.1f} K")
c3.metric("Rotation Speed", f"{rotational_speed} rpm")
c4.metric("Tool Wear", f"{tool_wear} min")

st.divider()

left, right = st.columns([1.05, 1])

with left:
    st.subheader("🔧 Machine Sensor Readings")
    st.dataframe(input_data, use_container_width=True, hide_index=True)

    chart_data = input_data.melt(
        var_name="Sensor",
        value_name="Reading"
    )
    fig = px.bar(
        chart_data,
        x="Sensor",
        y="Reading",
        color="Sensor",
        title="Current Sensor Values"
    )
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=90),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🤖 AI Failure Analysis")
    st.write("Run the trained Random Forest model on the selected readings.")

    if st.button("🔍 Predict Machine Failure", use_container_width=True):
        prediction = int(model.predict(input_data)[0])

        failure_probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            failure_index = list(model.classes_).index(1)
            failure_probability = float(probabilities[failure_index])

        if prediction == 1:
            st.error("⚠️ Potential Machine Failure Detected")
            st.write(
                "Recommended action: inspect the machine and review its "
                "maintenance history."
            )
        else:
            st.success("✅ No Machine Failure Predicted")
            st.write(
                "Continue routine inspections and follow the normal "
                "preventive maintenance schedule."
            )

        if failure_probability is not None:
            st.metric(
                "Model-estimated Failure Probability",
                f"{failure_probability:.2%}"
            )
            st.progress(failure_probability)

        st.caption(
            "Prediction is an estimate from a dataset-trained model, "
            "not a guarantee of real-world machine safety."
        )

st.divider()
st.subheader("📈 Model Performance")

if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    m2.metric("Failure Precision", f"{metrics.get('failure_precision', 0):.2%}")
    m3.metric("Failure Recall", f"{metrics.get('failure_recall', 0):.2%}")
    m4.metric("Failure F1-score", f"{metrics.get('failure_f1', 0):.2%}")

    st.info(
        "Failure recall is especially important: the model can miss some "
        "actual failures. Use this dashboard as an educational prototype, "
        "not as the sole basis for industrial safety decisions."
    )
else:
    st.warning("Model metrics file not found.")

st.divider()
st.caption(
    "Predictive Maintenance AI | Python • Scikit-learn • Streamlit • Plotly"
)
