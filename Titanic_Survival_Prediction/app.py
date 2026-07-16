# ==========================================================
# Import Required Libraries
# ==========================================================

# Streamlit is used to build the web application.
import streamlit as st

# Pandas is used to create the input DataFrame for prediction.
import pandas as pd

# Joblib is used to load the trained Decision Tree model.
import joblib

# Path is used to build file paths that work across operating systems.
from pathlib import Path
# ==========================================================
# Configure Streamlit Page
# ==========================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* ==========================================================
Main App
========================================================== */

.stApp{
    background-color:#F7FAFC;
}


/* ==========================================================
Main Container
========================================================== */

.block-container{
    max-width:1100px;
    padding-top:2rem;
    padding-bottom:2rem;
}


/* ==========================================================
Title
========================================================== */

h1{
    color:#0F4C81;
    text-align:center;
    font-weight:700;
}

p{
    font-size:17px;
}


/* ==========================================================
Sidebar
========================================================== */

section[data-testid="stSidebar"]{
    background-color:#EAF4FF;
}


/* ==========================================================
Input Fields
========================================================== */

div[data-baseweb="input"] > div{
    background-color:#EEF6FF !important;
    border:1px solid #BFD7ED !important;
    border-radius:10px !important;
}


/* ==========================================================
Select Boxes
========================================================== */

div[data-baseweb="select"] > div{
    background-color:#EEF6FF !important;
    border:1px solid #BFD7ED !important;
    border-radius:10px !important;
}


/* ==========================================================
Hover Effect
========================================================== */

div[data-baseweb="input"] > div:hover,
div[data-baseweb="select"] > div:hover{

    border:1px solid #1565C0 !important;

}


/* ==========================================================
Focus Effect
========================================================== */

div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div{

    border:2px solid #1565C0 !important;
    box-shadow:0 0 8px rgba(21,101,192,0.25);

}


/* ==========================================================
Button
========================================================== */

.stButton > button{

    background-color:#0F4C81;
    color:white;

    width:100%;
    height:52px;

    border:none;
    border-radius:10px;

    font-size:18px;
    font-weight:bold;

    transition:0.3s;

}


.stButton > button:hover{

    background-color:#1565C0;
    color:white;

}


/* ==========================================================
Success Message
========================================================== */

div[data-testid="stSuccess"]{

    border-radius:10px;

}


/* ==========================================================
Error Message
========================================================== */

div[data-testid="stError"]{

    border-radius:10px;

}


/* ==========================================================
DataFrame
========================================================== */

div[data-testid="stDataFrame"]{

    border-radius:10px;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Project Paths
# ==========================================================

# Get the directory where app.py is located.
BASE_DIR = Path(__file__).resolve().parent

# Path to the saved Decision Tree model.
MODEL_PATH = BASE_DIR / "Model" / "Decision_tree_model.pkl"

# ==========================================================
# Load Trained Machine Learning Model
# ==========================================================

@st.cache_resource
def load_model():
    """
    Load the trained Decision Tree model.

    Returns:
        Trained machine learning model.
    """
    return joblib.load(MODEL_PATH)


# Load the model once and reuse it throughout the application.
model = load_model()

# ==========================================================
# Sidebar
# ==========================================================

# The sidebar displays project information and usage instructions.
with st.sidebar:

    # Sidebar title
    st.title("Titanic Survival Prediction")

    # Short project description
    st.markdown("""
    This web application predicts whether a passenger
    would survive the Titanic disaster using a trained
    **Decision Tree Classifier**.
    """)

    st.divider()

    # Display feature information for users
    st.subheader("Input Features")

    st.markdown("""
    - Passenger Class
    - Gender
    - Age
    - Fare
    - Siblings / Spouses
    - Parents / Children
    - Embarked Port
    """)

    st.divider()

    # Display author information
    st.caption("Developed by")
    st.write("**Junaid Ur Rehman**")
    st.link_button(
    "🔗 View GitHub Repository",
    "https://github.com/junaidrehman41/ML_Projects"
)

# ==========================================================
# Header
# ==========================================================

col1, col2 = st.columns([1, 5], vertical_alignment="center")

with col1:
    st.image(
        "images/titanic_banner.png",
        width=120
    )

with col2:
    st.title("Titanic Survival Prediction")

    st.write(
        "Predict whether a passenger would survive the Titanic disaster "
        "using a trained **Decision Tree Classifier**."
    )


st.markdown(
"""
<div style="
background:#EAF4FF;
padding:15px 20px;
border-radius:12px;
border-left:6px solid #0F4C81;
margin-top:10px;
margin-bottom:20px;
">

<h3 style="margin:0;color:#0F4C81;">
Passenger Information
</h3>

<p style="margin-top:8px;">
Enter the passenger details below to predict whether the passenger would survive the Titanic disaster.
</p>

</div>
""",
unsafe_allow_html=True
)
# ==========================================================
# Create Two Columns
# ==========================================================

# Divide the page into two equal columns.
col1, col2 = st.columns(2)
# ==========================================================
# Left Column Inputs
# ==========================================================

with col1:

    # Passenger class selection
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3]
    )

    # Passenger gender
    sex = st.selectbox(
        "Gender",
        options=["Male", "Female"]
    )

    # Passenger age
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=25
    )

    # Passenger fare
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=32.20
    )
# ==========================================================
# Right Column Inputs
# ==========================================================

with col2:

    # Number of siblings/spouses aboard.
    sibsp = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )

    # Number of parents/children aboard.
    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )

    # Passenger embarkation port.
    embarked = st.selectbox(
        "Embarked",
        options=["Southampton", "Cherbourg", "Queenstown"]
    )
# ==========================================================
# Encode User Input
# ==========================================================

# Convert Gender into numerical values.
# Male   -> 0
# Female -> 1
sex = 0 if sex == "Male" else 1


# Convert Embarked into numerical values.
# Southampton -> 0
# Cherbourg   -> 1
# Queenstown  -> 2
embarked = {
    "Southampton": 0,
    "Cherbourg": 1,
    "Queenstown": 2
}[embarked]

# ==========================================================
# Prediction Button
# ==========================================================

# Create a button.
# Prediction starts only when this button is clicked.
predict_btn = st.button(
    "Predict Survival",
    use_container_width=True
)

# ==========================================================
# Make Prediction
# ==========================================================

if predict_btn:

    # Create a DataFrame containing the user inputs.
    input_data = pd.DataFrame({

        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]

    })

    # Predict the passenger's survival.
    prediction = model.predict(input_data)[0]

    # Calculate prediction probabilities.
    probability = model.predict_proba(input_data)

    confidence = probability.max() * 100

# ==========================================================
# Display Prediction
# ==========================================================

    st.divider()

    st.markdown("## 🚢 Prediction Result")

    if prediction == 1:

        st.success(
            f"""
            Passenger is likely to **Survive**

            Confidence : **{confidence:.2f}%**
            """
        )

    else:

        st.error(
            f"""
            Passenger is likely to **Not Survive**

            Confidence : **{confidence:.2f}%**
            """
        )

# ==========================================================
# Display User Inputs
# ==========================================================

    with st.expander("View Entered Passenger Details"):

        st.dataframe(input_data)

st.divider()

st.markdown("""
---
### Developed by

**Junaid Ur Rehman**

📧 Junaidurrehman41@gmail.com

🐙 https://github.com/junaidrehman41
""")
