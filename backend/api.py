from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import psycopg2
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

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

# load the model
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

# define object for prediction
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

# prediction function
@app.post("/predict")
def predict(car: CarInput):
    # create row from the user input data
    row = pd.DataFrame([{k: getattr(car, k) for k in FEATURES}])

    # predict and exponentiate the logged value
    pred_log = model.predict(row)
    pred = float(np.exp(pred_log)[0])

    # log data into db
    with conn.cursor() as cur:
        # insert user input
        cur.execute(
            """
            INSERT INTO user_input_data(
                age, manufacturer, model, trim, condition, fuel,
                odometer, drive, vehicle_type, color, state, lat, long
            )
            VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """, (
                car.age,
                car.manufacturer,
                car.model,
                car.trim,
                car.condition,
                car.fuel,
                car.odometer,
                car.drive,
                car.type,
                car.paint_color,
                car.state,
                car.lat,
                car.long
            )
        )

        # get the input ID 
        input_id = cur.fetchone()[0]

        # insert the prediction
        cur.execute("""
                INSERT INTO predictions (
                    input_id, predicted_price, model_version)
                    VALUES (%s,%s,%s);
            """, (
                input_id,
                pred,
                "random_forest_v1"
            ))

    return {"predicted_price": pred}