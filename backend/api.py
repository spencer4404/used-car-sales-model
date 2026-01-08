from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import psycopg2
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (fine for demo)
    allow_methods=["*"],
    allow_headers=["*"],
)

# connect to database
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL is None:
    raise RuntimeError("DATABASE_URL is not set!")
else:
    print(DATABASE_URL)

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True
print("Connected to Postgres")

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