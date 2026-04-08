from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import uvicorn

app = FastAPI(title="Antibiotic Resistance Forecaster API")

# Setup paths (relative to project root where we will run the server)
MODEL_DIR = 'models'
TRAINED_DIR = os.path.join(MODEL_DIR, 'trained')

try:
    xgb_model = joblib.load(os.path.join(TRAINED_DIR, 'xgboost.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
except Exception as e:
    print("Warning: Models / preprocessors not found. Ensure models are trained before using the API.")
    # Initialize as None to allow app to start, but fail gracefully on requests
    xgb_model, scaler, encoders, feature_names = None, None, None, None

class PatientData(BaseModel):
    Age: float
    Gender: str
    Region: str
    Hospital_ID: str
    Infection_Type: str
    Bacterial_Strain: str
    Antibiotic: str
    Dosage_mg: float
    Duration_days: float
    # Provide these if available, otherwise we will default them
    Rolling_Region_Res_Rate_30d: float = 0.0
    Rolling_Hosp_Abx_Usage_30d: float = 0.0
    Month: int = 1
    DayOfWeek: int = 0
    Year: int = 2023

@app.get("/")
def home():
    return {"message": "Antibiotic Resistance Forecaster API is running"}

@app.post("/predict")
def predict_resistance(data: PatientData):
    if not xgb_model:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    # Convert to dataframe
    df = pd.DataFrame([data.dict()])
    
    # Process categories
    categorical_cols = ['Gender', 'Region', 'Hospital_ID', 'Infection_Type', 'Bacterial_Strain', 'Antibiotic']
    for col in categorical_cols:
        if col in encoders:
            le = encoders[col]
            val = str(df[col].iloc[0])
            if val in le.classes_:
                df[col] = le.transform([val])[0]
            else:
                # default to 0 if unseen
                df[col] = 0
                
    # Ensure correct order of features
    X_input = df[feature_names]
    X_scaled = scaler.transform(X_input)
    
    # Predict
    prob = float(xgb_model.predict_proba(X_scaled)[0, 1])
    
    res = {
        "resistance_probability": round(prob, 4),
        "risk_level": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low",
        "alert": "Warning: High likelihood of antibiotic resistance. Consider alternative treatments or cultures." if prob > 0.6 else "Normal risk."
    }
    
    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
