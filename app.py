import streamlit as st
import joblib

model = joblib.load('model/salary_model.pkl')

st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

st.title("💰 Salary Prediction App")
st.write("Enter years of experience below to predict expected salary.")

st.divider()

experience = st.slider('Years of Experience', min_value=0.0, max_value=50.0, value=1.0, step=0.1)

if st.button('🔍 Predict Salary', use_container_width=True):
    prediction = model.predict([[experience]])
    st.success(f'💵 Predicted Salary: **${prediction[0]:,.2f}**')