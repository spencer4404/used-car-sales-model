from fastapi import FastAPI, HTTPException
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
conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True

# load the model
model = joblib.load("modelv1.joblib")

SMEAR = 1.0129192399257083

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
    # Row must match modelv1.joblib's expected columns exactly:
    # cat: manufacturer, model, trim, condition, fuel, drive, type,
    #      paint_color, state; numeric: age, odometer, lat, long.
    # (Extra columns are dropped, but missing ones raise ValueError.)
    trim = (car.trim or "").strip().lower() or "unknown"
    try:
        row = pd.DataFrame([{
            'age': car.age,
            'manufacturer': car.manufacturer.lower(),
            'model': car.model.lower(),
            'trim': trim,
            'condition': car.condition.lower(),
            'fuel': car.fuel.lower(),
            'odometer': car.odometer,
            'drive': car.drive.lower(),
            'type': car.type.lower(),
            'paint_color': car.paint_color.lower(),
            'state': car.state.lower(),
            'lat': car.lat,
            'long': car.long,
        }])

        # predict and exponentiate the logged value
        pred_log = model.predict(row)
    except HTTPException:
        raise
    except Exception as e:
        # Convert to HTTPException so CORSMiddleware still attaches
        # CORS headers (unhandled 500s surface in browsers as CORS errors).
        raise HTTPException(status_code=500, detail=f"prediction failed: {e}")
    pred = float(np.exp(pred_log)[0] * SMEAR)

    # log data into db
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"db logging failed: {e}")

    return {"predicted_price": pred}


@app.get("/health")
def health():
    return {"ok": True}