import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Define Input Data Model ---
class AirfareFeatures(BaseModel):
    # Numerical features
    nsmiles: float = Field(..., example=1589.0)
    passengers: float = Field(..., example=229.0)
    large_ms: float = Field(..., example=0.6179)
    lf_ms: float = Field(..., example=0.3444)
    competition_ratio: float = Field(..., example=1.794)
    
    # --- THIS IS THE FIX ---
    # Change Year and quarter BACK to int
    Year: int = Field(..., example=2023)
    quarter: int = Field(..., example=3)
    # --- END OF FIX ---

    # Categorical features
    carrier_lg: str = Field(..., example="AA")
    carrier_low: str = Field(..., example="DL")
    airport_1: str = Field(..., example="JFK")
    airport_2: str = Field(..., example="LAX")
    route: str = Field(..., example="JFK_LAX")
    year_quarter: str = Field(..., example="2023_Q3")

# --- 2. Define Output Data Model ---
class PredictionOut(BaseModel):
    predicted_fare: float

# --- 3. Create FastAPI app ---
app = FastAPI(
    title="Airfare Price Prediction API",
    description="API for predicting U.S. airfare using an XGBoost model."
)

# --- 4. Add CORS Middleware ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. Load the Model ---
try:
    model = joblib.load('airfare_model_pipeline.joblib')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model file not found. Make sure 'airfare_model_pipeline.joblib' is in the same directory.")
    model = None

# --- 6. Create the Prediction Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Airfare Prediction API. Go to /docs for more info."}

@app.post("/predict", response_model=PredictionOut)
def predict_fare(features: AirfareFeatures):
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}

    input_data = pd.DataFrame([features.model_dump()])
    prediction = model.predict(input_data)[0]
    
    return {"predicted_fare": round(float(prediction), 2)}