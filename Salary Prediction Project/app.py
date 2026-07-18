import streamlit as st
import joblib
import os

# -----------------------------
# Page config
# This must be the very first Streamlit command in the script
# -----------------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

# -----------------------------
# Custom styling
# A soft background gradient and a styled card for the prediction form,
# kept subtle so it stays professional rather than flashy
# -----------------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(180deg, #f4f9ff 0%, #eef2f7 100%);
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #0f2540;
    }
    section[data-testid="stSidebar"] * {
        color: #f0f4f8 !important;
    }

    /* Main title */
    h1 {
        color: #0f2540;
    }

    /* Predict button */
    div.stButton > button {
        background-color: #0f2540;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 0;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #16375e;
        color: white;
    }

    /* Success message box */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Get the folder where this script (app.py) is located
# This ensures file paths work correctly whether the app runs
# locally or on Streamlit Cloud (where the working directory differs)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "salary_model.pkl")

# -----------------------------
# Load the trained model
# Wrapped in try/except so the app shows a friendly message
# instead of crashing if the model file is missing
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# -----------------------------
# Sidebar: project info + author
# Keeps the main page focused on the prediction form,
# while extra context sits neatly on the side
# -----------------------------
with st.sidebar:
    st.header("About This App")
    st.write(
        "Predicts an employee's expected **salary** based on their "
        "years of experience, using a trained **Linear Regression** model."
    )
    st.divider()
    st.subheader("Input Features")
    st.markdown("- Years of Experience")
    st.divider()
    st.subheader("Author")
    st.markdown("**Junaid Ur Rehman**")
    st.markdown("[GitHub](https://github.com/junaidrehman41)")
    st.markdown("junaidrehman41@gmail.com")
    st.divider()
    st.caption("Built with Streamlit & scikit-learn")

# -----------------------------
# Header
# -----------------------------
st.title("💰 Salary Prediction App")
st.write("Enter years of experience below to predict expected salary.")
st.divider()

# -----------------------------
# Stop early with a clear message if the model failed to load,
# instead of letting the app crash further down
# -----------------------------
if not model_loaded:
    st.error("⚠️ Model file not found. Please make sure 'model/salary_model.pkl' exists.")
    st.stop()

# -----------------------------
# Input: Years of Experience
# -----------------------------
experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.1
)

st.divider()

# -----------------------------
# Predict button
# -----------------------------
if st.button("🔍 Predict Salary", use_container_width=True):
    prediction = model.predict([[experience]])
    st.success(f"💵 Predicted Salary: **${prediction[0]:,.2f}**")