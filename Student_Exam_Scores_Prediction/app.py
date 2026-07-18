import streamlit as st
import joblib
import numpy as np

# Page config (yeh sabse upar, imports ke turant baad honi chahiye)
st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓", layout="centered")

# Load model & scaler
model = joblib.load("Model/logistic_regression_model.pkl")
scaler = joblib.load("Model/scaler.pkl")

# -----------------------------
# Sidebar: project info + author
# Keeps the main page focused on the prediction form,
# while extra context sits neatly on the side
# -----------------------------
with st.sidebar:
    st.header("About This App")
    st.write(
        "Predicts whether a student will **Pass** or **Fail** "
        "based on study habits and academic history, using a "
        "trained **Logistic Regression** model."
    )
    st.divider()
    st.subheader("Input Features")
    st.markdown(
        "- Hours Studied\n"
        "- Sleep Hours\n"
        "- Attendance (%)\n"
        "- Previous Scores"
    )
    st.divider()
    st.subheader("Author")
    st.markdown("**Junaid Ur Rehman**")
    st.markdown("[GitHub](https://github.com/junaidrehman41)")
    st.markdown("junaidrehman41@gmail.com")
    st.divider()
    st.caption("Built with Streamlit & scikit-learn")

# Header
st.title("🎓 Student Exam Pass/Fail Predictor")
st.markdown("Enter student details below to predict whether they'll **Pass** or **Fail**.")
st.divider()

# Inputs in two columns (better layout)
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input("📚 Hours Studied", min_value=0.0, max_value=24.0, value=5.0)
    attendance_percent = st.number_input("📅 Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)

with col2:
    sleep_hours = st.number_input("😴 Sleep Hours", min_value=0.0, max_value=12.0, value=7.0)
    previous_scores = st.number_input("📝 Previous Scores", min_value=0.0, max_value=100.0, value=60.0)

st.divider()

# Predict button (centered-ish, full width)
if st.button("🔮 Predict Result", use_container_width=True):
    input_data = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"### ✅ Prediction: PASS")
        st.metric("Confidence", f"{probability:.1%}")
    else:
        st.error(f"### ❌ Prediction: FAIL")
        st.metric("Confidence", f"{1-probability:.1%}")