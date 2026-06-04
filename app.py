import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# ── Load Model ────────────────────────────────────────────────
model = joblib.load("student_model.pkl")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])

if theme == "Dark":
    st.markdown("""
    <style>
    .stApp { background-color:#0E1117; color:white; }
    </style>
    """, unsafe_allow_html=True)

# ── Model Comparison (Sidebar) ────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Model Comparison")
st.sidebar.caption("Dataset: Synthetically generated based on educational research patterns. "
                   "Real student data was avoided due to privacy concerns.")

if os.path.exists("model_results.csv"):
    results_df = pd.read_csv("model_results.csv")
    st.sidebar.dataframe(results_df, hide_index=True)
    st.sidebar.success("✅ Best Model: Gradient Boosting (R²: 0.8259)")
else:
    st.sidebar.info("Run train_model.py to generate comparison results.")

# ── Title ─────────────────────────────────────────────────────
st.title("🎓 Student Performance Prediction System")
st.write("Enter student details below to predict exam performance and mental wellness.")

# ── Student Info ──────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    name = st.text_input("👤 Student Name")
with col2:
    student_id = st.text_input("🆔 Student ID")

st.markdown("---")

# ── Input Sliders ─────────────────────────────────────────────
st.subheader("📋 Academic & Behavioral Factors")

col1, col2 = st.columns(2)
with col1:
    study_hours        = st.slider("📖 Study Hours",              0, 12, 4)
    sleep_hours        = st.slider("😴 Sleep Hours",              0, 12, 7)
    stress_level       = st.slider("😰 Stress Level",             1,  5, 3)
    attendance         = st.slider("🏫 Attendance (%)",           0,100,80)
    social_media_usage = st.slider("📱 Social Media Usage (hrs)", 0, 12, 2)

with col2:
    focus_level        = st.slider("🎯 Focus Level",              1, 5, 3)
    motivation_level   = st.slider("💪 Motivation Level",         1, 5, 3)
    revision_frequency = st.slider("🔁 Revision Frequency",       1, 5, 3)
    concentration_level= st.slider("🧘 Concentration Level",      1, 5, 3)
    class_attention    = st.slider("👁️ Class Attention",           1, 5, 3)

st.markdown("---")

# ── Mental Wellness Analysis ──────────────────────────────────
st.subheader("🧠 Mental Wellness Analysis")

mental_score = (
    (stress_level * 2)
    + (social_media_usage * 0.5)
    - (sleep_hours * 0.5)
    - motivation_level
)

if mental_score >= 7:
    mental_state = "High Stress"
    st.error("⚠️ High Stress Level Detected")
elif mental_score >= 3:
    mental_state = "Moderate Stress"
    st.warning("⚠️ Moderate Stress Level Detected")
else:
    mental_state = "Healthy"
    st.success("😊 Healthy Mental State")

st.write(f"**Mental State:** {mental_state}")

wellness_score = max(0, min(100, int(
    100
    - (stress_level * 15)
    - (social_media_usage * 3)
    + (sleep_hours * 5)
    + (motivation_level * 5)
)))

st.subheader("📊 Mental Wellness Score")
st.progress(wellness_score)
st.write(f"**Wellness Score: {wellness_score}/100**")

# ── Mental Health Suggestions ─────────────────────────────────
st.subheader("💡 Mental Health Suggestions")

suggestions = []
if stress_level >= 4:
    suggestions.append("🧘 Try meditation or deep breathing to reduce stress.")
if sleep_hours < 6:
    suggestions.append("😴 Aim for at least 7–8 hours of sleep for better focus.")
if social_media_usage > 4:
    suggestions.append("📵 Limit social media to under 2 hours per day.")
if motivation_level <= 2:
    suggestions.append("🎯 Set small daily goals to rebuild motivation.")
if revision_frequency <= 2:
    suggestions.append("📖 Revise notes at least 3 times a week for retention.")

if suggestions:
    for s in suggestions:
        st.info(s)
else:
    st.success("✅ Great habits! Keep maintaining this balance.")

st.markdown("---")

# ── Predict Button ────────────────────────────────────────────
if st.button("🔮 Predict Marks", use_container_width=True):

    input_data = np.array([[
        study_hours, sleep_hours, stress_level, attendance,
        focus_level, motivation_level, social_media_usage,
        revision_frequency, concentration_level, class_attention
    ]])

    predicted_marks = model.predict(input_data)[0]
    predicted_marks = round(float(np.clip(predicted_marks, 0, 100)), 2)

    # ── Result ────────────────────────────────────────────────
    st.markdown("---")
    if name:
        st.subheader(f"📋 Results for {name} (ID: {student_id})")

    st.success(f"📚 Predicted Marks: **{predicted_marks:.2f}%**")

    # ── Success Meter ─────────────────────────────────────────
    st.subheader("🎯 Success Meter")
    st.progress(int(predicted_marks))

    # ── Grade ─────────────────────────────────────────────────
    st.subheader("🏆 Grade Prediction")
    if predicted_marks >= 90:
        grade = "A+"
    elif predicted_marks >= 80:
        grade = "A"
    elif predicted_marks >= 70:
        grade = "B"
    elif predicted_marks >= 60:
        grade = "C"
    else:
        grade = "D"

    grade_colors = {"A+": "🟢", "A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}
    st.write(f"**Grade: {grade_colors.get(grade, '')} {grade}**")

    # ── Performance Analysis ──────────────────────────────────
    st.subheader("📊 Performance Analysis")
    if predicted_marks >= 85:
        st.balloons()
        st.success("🌟 Excellent Performance!")
    elif predicted_marks >= 70:
        st.info("👍 Good Performance")
    elif predicted_marks >= 50:
        st.warning("⚠️ Average Performance — some areas need work")
    else:
        st.error("📚 Needs Improvement — focus on study hours & attendance")

    # ── Performance Bar Chart ─────────────────────────────────
    st.subheader("📈 Performance Factors Breakdown")

    factors = {
        "Study Hours":    study_hours,
        "Sleep":          sleep_hours,
        "Attendance":     attendance / 10,   # scaled to /10 for visibility
        "Focus":          focus_level * 2,
        "Motivation":     motivation_level * 2,
        "Concentration":  concentration_level * 2,
        "Class Attention":class_attention * 2,
    }

    colors = ["#4CAF50" if v >= 6 else "#FF9800" if v >= 3 else "#F44336"
              for v in factors.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(list(factors.keys()), list(factors.values()), color=colors, edgecolor="white")
    ax.set_ylabel("Score (scaled)")
    ax.set_title("Your Performance Factors", fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("🟢 Strong  🟠 Moderate  🔴 Needs Work")

    # ── Feature Importance ────────────────────────────────────
    st.subheader("🔍 What Impacts Your Marks Most?")

    feature_names = [
        "Study Hours", "Sleep Hours", "Stress Level", "Attendance",
        "Focus Level", "Motivation", "Social Media", "Revision",
        "Concentration", "Class Attention"
    ]
    importances = model.feature_importances_

    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.barh(imp_df["Feature"], imp_df["Importance"], color="#2196F3")
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Feature Importance (Gradient Boosting)", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")
    st.caption("🤖 Model: Gradient Boosting Regressor | R² Score: 0.8259 | MAE: 4.18 marks")
    st.caption("📊 Dataset: Synthetically generated based on educational research patterns. "
               "Real student data avoided due to privacy concerns.")