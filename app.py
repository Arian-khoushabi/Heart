import os
import streamlit as st
import pandas as pd
import joblib
import cohere
from model_preprocess import HeartModel

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.caption(
    "Machine-learning cardiovascular risk classification "
    "with AI-assisted clinical interpretation."
)
st.markdown("*Created by Arian Khoushabi*")

@st.cache_resource
def load_heart_model():
    return joblib.load("Heart_ai.pkl")


try:
    model: HeartModel = load_heart_model()

except Exception as e:
    st.error(f"Could not load Heart_ai.pkl: {e}")
    st.stop()

def get_cohere_api_key():

    try:
        return st.secrets["COHERE_API_KEY"]

    except Exception:
        return os.getenv("COHERE_API_KEY")


COHERE_API_KEY = get_cohere_api_key()


@st.cache_resource
def get_cohere_client(api_key):
    return cohere.ClientV2(api_key=api_key)

def generate_clinical_interpretation(
    risk_label,
    probability,
    patient_data
):

    if not COHERE_API_KEY:
        return (
            "Cohere interpretation is unavailable because "
            "COHERE_API_KEY has not been configured."
        )

    client = get_cohere_client(COHERE_API_KEY)

    system_prompt = """
You are a clinical decision-support assistant.

Your role is to provide a concise professional interpretation
of the output of a machine-learning heart-disease prediction model
for a healthcare professional.

You MAY:
- Explain the model's predicted risk classification.
- Explain the predicted probability.
- Summarize clinically relevant patient parameters.
- Identify general cardiovascular patterns represented by the inputs.
- Explain why the result may warrant clinical attention.
- Describe general categories of further clinical evaluation.

You MUST NOT:
- Diagnose heart disease.
- Claim the machine-learning prediction is a confirmed diagnosis.
- Prescribe medications.
- Recommend medication changes.
- Replace clinical judgment.
- Claim causality from individual input variables.
- Present the probability as a validated real-world disease probability.

Important:
The machine-learning model was developed from a public heart-disease
research dataset and has not been established as a clinically validated
diagnostic system.

Use concise, objective, medically professional language.

End with a short limitations statement.
"""

    user_prompt = f"""
MODEL OUTPUT

Risk classification: {risk_label}
Model probability: {probability:.1%}

PATIENT PARAMETERS

Age: {patient_data['Age']}
Sex: {patient_data['Sex_label']}
Chest pain type: {patient_data['ChestPainType_label']}
Resting blood pressure: {patient_data['RestingBP']} mmHg
Cholesterol: {patient_data['Cholesterol']} mg/dL
Fasting blood sugar > 120 mg/dL: {patient_data['FastingBS_label']}
Resting ECG: {patient_data['RestingECG_label']}
Maximum heart rate: {patient_data['MaxHR']} bpm
Exercise-induced angina: {patient_data['ExerciseAngina_label']}
Oldpeak / ST depression: {patient_data['Oldpeak']}

Provide a concise interpretation suitable for a healthcare professional.

Separate the response into:

1. Model interpretation
2. Relevant clinical features
3. Clinical context
4. Limitations
"""

    try:

        response = client.chat(
            model="command-a-plus-05-2026",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.3
        )

        text_parts = []

        for content in response.message.content:
            if getattr(content, "type", None) == "text":
                text_parts.append(content.text)

        if not text_parts:
            return "Cohere returned no textual interpretation."

        return "\n".join(text_parts)

    except Exception as e:
        return (
            "AI interpretation is currently unavailable. "
            f"Cohere error: {e}"
        )

st.subheader("Patient Data")

with st.form("heart_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        Age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        Sex = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

        ChestPainType = st.selectbox(
            "Chest Pain Type",
            [
                "TA (Typical Angina)",
                "ATA (Atypical Angina)",
                "NAP (Non-Anginal Pain)",
                "ASY (Asymptomatic)"
            ]
        )

        RestingECG = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST",
                "LVH"
            ]
        )

        ExerciseAngina = st.selectbox(
            "Exercise Angina",
            [
                "No",
                "Yes"
            ]
        )

    with col2:

        RestingBP = st.number_input(
            "Resting Blood Pressure (mmHg)",
            min_value=60,
            max_value=250,
            value=120
        )

        Cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=200
        )

        FastingBS = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [
                "No",
                "Yes"
            ]
        )

        MaxHR = st.number_input(
            "Maximum Heart Rate",
            min_value=60,
            max_value=250,
            value=150
        )

        Oldpeak = st.number_input(
            "Oldpeak (ST Depression)",
            min_value=-2.0,
            max_value=10.0,
            value=0.0,
            step=0.1
        )

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True
    )

if submitted:

    Sex_encoded = 1 if Sex == "Male" else 0

    chest_pain_code = {
        "TA (Typical Angina)": 0,
        "ATA (Atypical Angina)": 1,
        "NAP (Non-Anginal Pain)": 2,
        "ASY (Asymptomatic)": 3
    }

    resting_ecg_code = {
        "Normal": 0,
        "ST": 1,
        "LVH": 2
    }

    ChestPainType_encoded = chest_pain_code[ChestPainType]
    RestingECG_encoded = resting_ecg_code[RestingECG]

    FastingBS_encoded = (
        1 if FastingBS == "Yes" else 0
    )

    ExerciseAngina_encoded = (
        1 if ExerciseAngina == "Yes" else 0
    )

    model_data = {
        "Age": float(Age),
        "Sex": Sex_encoded,
        "ChestPainType": ChestPainType_encoded,
        "RestingBP": float(RestingBP),
        "Cholesterol": float(Cholesterol),
        "FastingBS": FastingBS_encoded,
        "RestingECG": RestingECG_encoded,
        "MaxHR": float(MaxHR),
        "ExerciseAngina": ExerciseAngina_encoded,
        "Oldpeak": float(Oldpeak)
    }

    input_df = pd.DataFrame([model_data])

    try:

        prediction = int(
            model.predict(input_df)[0]
        )

        probability = float(
            model.predict_proba(input_df)[0][1]
        )

        risk_label = (
            "HIGH RISK"
            if prediction == 1
            else "LOW RISK"
        )

    except Exception as e:

        st.error(
            f"Model prediction failed: {e}"
        )

        st.stop()

    st.divider()

    st.subheader("Prediction Result")

    if risk_label == "HIGH RISK":

        st.error(
            f"Risk Classification: {risk_label}"
        )

    else:

        st.success(
            f"Risk Classification: {risk_label}"
        )


    probability_percent = probability * 100

    st.metric(
        label="Model Probability",
        value=f"{probability_percent:.2f}%"
    )

    st.progress(
        min(
            max(probability, 0.0),
            1.0
        )
    )



    clinical_data = {
        **model_data,
        "Sex_label": Sex,
        "ChestPainType_label": ChestPainType,
        "RestingECG_label": RestingECG,
        "FastingBS_label": FastingBS,
        "ExerciseAngina_label": ExerciseAngina
    }

    st.subheader("🧠 AI Clinical Interpretation")

    with st.spinner(
        "Generating clinical interpretation..."
    ):

        interpretation = generate_clinical_interpretation(
            risk_label=risk_label,
            probability=probability,
            patient_data=clinical_data
        )

    st.info(interpretation)

    st.warning(
        "A note from Arian: this model has not been clinically "
        "validated for diagnosis or treatment decisions. "
        "Its output should not be used as a substitute for "
        "professional clinical assessment."
    )