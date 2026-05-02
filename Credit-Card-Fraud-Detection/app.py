import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="💳 Fraud Detection", page_icon="💳", layout="wide")

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #eaeef3;
}
.header {
    padding: 16px;
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(8px);
    margin-bottom: 10px;
}
.card {
    padding: 16px;
    border-radius: 16px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(8px);
}
.result-normal { border-left: 6px solid #00ffa3; }
.result-fraud { border-left: 6px solid #ff4b5c; }
.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
bundle = joblib.load("models/fraud_model.pkl")
model = bundle["model"]
scaler = bundle["scaler"]
required_features = scaler.n_features_in_

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="header">
<h2>💳 Credit Card Fraud Detection</h2>
<p>Real-time ML scoring with charts & explainability</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Controls")

threshold = st.sidebar.slider("Fraud Threshold", 0.1, 0.9, 0.5, 0.05)

if st.sidebar.button("⚡ Load NORMAL Sample"):
    st.session_state["sample"] = "normal"

if st.sidebar.button("🚨 Load FRAUD Sample"):
    st.session_state["sample"] = "fraud"

# -----------------------------
# INPUT GRID
# -----------------------------
st.subheader("🧾 Enter Transaction Features")

cols = st.columns(5)
inputs = []

for i in range(required_features):
    default = 0.0

    if "sample" in st.session_state:
        if st.session_state["sample"] == "fraud":
            default = np.random.uniform(-5, 5)  # extreme values
        else:
            default = np.random.uniform(-0.2, 0.2)

    with cols[i % 5]:
        val = st.number_input(f"V{i}", value=float(default), format="%.4f")
        inputs.append(val)

# -----------------------------
# PREDICT
# -----------------------------
if st.button("🚀 Predict Fraud"):

    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prob = float(model.predict_proba(X_scaled)[0][1])
    label = "FRAUD" if prob >= threshold else "NORMAL"

    st.markdown("---")

    # -----------------------------
    # RESULT CARD
    # -----------------------------
    if label == "FRAUD":
        st.markdown(f"""
        <div class="card result-fraud">
        <h3>🚨 FRAUD DETECTED</h3>
        <p>Risk Score: {prob:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card result-normal">
        <h3>✅ NORMAL TRANSACTION</h3>
        <p>Confidence: {1-prob:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # CHARTS SECTION
    # -----------------------------
    st.subheader("📊 Prediction Visuals")

    c1, c2 = st.columns(2)

    # -------- Gauge --------
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Fraud Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    # -------- Pie Chart --------
    with c2:
        fig = px.pie(
            names=["Fraud", "Normal"],
            values=[prob, 1 - prob],
            title="Fraud vs Normal Probability"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------- Bar Chart --------
    st.subheader("📊 Probability Breakdown")

    fig = px.bar(
        x=["Fraud", "Normal"],
        y=[prob, 1 - prob],
        color=["Fraud", "Normal"],
        title="Fraud vs Normal"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------- Feature Importance --------
    if hasattr(model, "feature_importances_"):
        st.subheader("📈 Feature Importance")

        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-10:]

        fig = px.bar(
            x=importance[top_idx],
            y=[f"V{i}" for i in top_idx],
            orientation='h',
            title="Top Contributing Features"
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using ML + FastAPI + Streamlit")