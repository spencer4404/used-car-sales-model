from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (fine for demo)
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("car_price_model.joblib")

# define the features
FEATURES = [
    'age',
    'manufacturer',
    'model',
    'trim',
    'condition',
    'fuel',
    'odometer',
    'drive',
    'type',
    'paint_color',
    'state',
    'lat',
    'long'
]

class CarInput(BaseModel):
    age: int
    manufacturer: str
    model: str
    trim: str
    condition: str
    fuel: str
    odometer: int
    drive: str
    type: str
    paint_color: str
    state: str
    lat: float
    long: float

@app.post("/predict")
def predict(car: CarInput):
    row = pd.DataFrame([{k: getattr(car, k) for k in FEATURES}])
    pred_log = model.predict(row)
    pred = float(np.exp(pred_log)[0])
    return {"predicted_price": pred}