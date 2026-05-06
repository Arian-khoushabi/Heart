from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from model_preprocess import HeartModel
from fastapi.middleware.cors import CORSMiddleware
import requests

# ---------------------------------------
# Initialize FastAPI
# ---------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# Load ML Model
# ---------------------------------------
try:
    model: HeartModel = joblib.load("Heart_ai.pkl")
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# ---------------------------------------
# GapGPT Credentials
# ---------------------------------------
GAPGPT_API_KEY = "sk-uE1NwqeXapyKJNOEUqyyumoxSekgMl3iokmHiUcdp2SR9mUy"
GAPGPT_URL = "https://api.gapgpt.app/v1/chat/completions"


# ---------------------------------------
# Function to Call GapGPT
# ---------------------------------------
def call_gapgpt(system_prompt, user_prompt):

    headers = {
        "Authorization": f"Bearer {GAPGPT_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 350,
        "temperature": 0.3
    }

    response = requests.post(GAPGPT_URL, json=payload, headers=headers, timeout=30)
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------
# Pydantic Input Schema
# ---------------------------------------
class HeartData(BaseModel):
    Age: float
    Sex: int
    ChestPainType: int
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    RestingECG: int
    MaxHR: float
    ExerciseAngina: int
    Oldpeak: float


# ---------------------------------------
# Prediction + LLM Endpoint
# ---------------------------------------
@app.post("/predict")
def predict_heart_disease(data: HeartData):
    try:
        input_df = pd.DataFrame([data.model_dump()])

        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # ---- PROMPTS ----
    system_prompt = """
    You are a clinical decision-support assistant. 
    Your role is to help doctors and surgeons interpret the output of a heart-disease risk prediction model.

    You MAY:
    - Explain the risk level indicated by the model.
    - Summarize how patient parameters may contribute to cardiovascular risk.
    - Highlight potentially relevant clinical patterns.
    - Describe general non-prescriptive categories of further evaluation.
    - Consider peri-operative relevance.

    You MUST NOT:
    - Give a diagnosis.
    - Prescribe medications or treatments.
    - Tell the clinician to order a test.
    - Override clinical judgment.
    - Use emotional or alarming language.

    Your tone must be:
    - concise
    - objective
    - medically professional
    - not patient-facing

    End with a short limitations statement.
    """

    user_prompt = f"""
    Risk classification: {risk_label}
    Probability: {probability:.1%}

    Patient parameters:
    Age: {data.Age}
    Sex: {data.Sex}
    ChestPainType: {data.ChestPainType}
    RestingBP: {data.RestingBP}
    Cholesterol: {data.Cholesterol}
    FastingBS: {data.FastingBS}
    RestingECG: {data.RestingECG}
    MaxHR: {data.MaxHR}
    ExerciseAngina: {data.ExerciseAngina}
    Oldpeak: {data.Oldpeak}

    Provide a concise interpretation suitable for a doctor or surgeon.
    """

    # ---------------------------------------
    # LLM CALL
    # ---------------------------------------
    try:
        interpretation = call_gapgpt(system_prompt, user_prompt)

    except Exception as e:
        print("LLM ERROR:", e)
        interpretation = (
            "Interpretation unavailable due to LLM error. "
            "Use model output with standard clinical judgment."
        )

    # ---------------------------------------
    # RETURN JSON
    # ---------------------------------------
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_label": risk_label,
        "interpretation": interpretation
    }

@app.get("/")
def root():
    return {"message": "Heart disease prediction API is running with LLM support!"}