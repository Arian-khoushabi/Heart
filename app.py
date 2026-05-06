import streamlit as st
import requests

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")

api_url = st.text_input("API URL", "http://127.0.0.1:8000/predict")

st.subheader("Enter Patient Data")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 1, 120)
    Sex = st.selectbox("Sex", ["Female", "Male"])
    ChestPainType = st.selectbox(
        "Chest Pain Type",
        ["TA (Typical Angina)", "ATA (Atypical Angina)", "NAP (Non-Anginal Pain)", "ASY (Asymptomatic)"]
    )
    RestingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    ExerciseAngina = st.selectbox("Exercise Angina", ["No", "Yes"])

with col2:
    RestingBP = st.number_input("Resting Blood Pressure", 60, 250)
    Cholesterol = st.number_input("Cholesterol", 100, 600)
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    MaxHR = st.number_input("Max Heart Rate", 60, 250)
    Oldpeak = st.number_input("Oldpeak (ST Depression)", -2.0, 10.0, step=0.1)

if st.button("Predict"):
    with st.spinner("Contacting model and generating interpretation..."):

        data = {
            "Age": Age,
            "Sex": 1 if Sex == "Male" else 0,
            "ChestPainType": ["TA", "ATA", "NAP", "ASY"].index(ChestPainType.split()[0]),
            "RestingBP": RestingBP,
            "Cholesterol": Cholesterol,
            "FastingBS": 1 if FastingBS == "Yes" else 0,
            "RestingECG": ["Normal", "ST", "LVH"].index(RestingECG),
            "MaxHR": MaxHR,
            "ExerciseAngina": 1 if ExerciseAngina == "Yes" else 0,
            "Oldpeak": Oldpeak
        }

        try:
            response = requests.post(api_url, json=data, timeout=40)
            res = response.json()
        except Exception as e:
            st.error(f"❌ Error connecting to API: {e}")
            st.stop()

        if "error" in res:
            st.error(f"API Error: {res['error']}")
            st.stop()

        st.subheader("Prediction Result")

        if res["risk_label"] == "HIGH RISK":
            st.error(f"Risk Level: {res['risk_label']}")
        else:
            st.success(f"Risk Level: {res['risk_label']}")

        probability_percent = round(res["probability"] * 100, 2)
        st.write(f"Probability: {probability_percent}%")

        st.progress(min(int(probability_percent), 100))

        st.subheader("🧠 AI Clinical Interpretation")
        st.info(res["interpretation"])