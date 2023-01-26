import logging

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from app.model.model import __version__ as model_version
from app.model.model import predict_power

app = FastAPI()

class DataIn(BaseModel):
    pitch_angle: float
    abs_wind_dir: float
    humidity: float
    pressure: float
    temp: float
    wind_speed: float
    nacelle_angle: float

class PredictedPower(BaseModel):
    power: float

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model = PredictedPower)
def predict(payload: DataIn):
    measurements = np.array([payload.pitch_angle, payload.abs_wind_dir, payload.humidity,
                    payload.pressure, payload.temp, payload.wind_speed,
                    payload.nacelle_angle])
    measurements = measurements.reshape((1, len(measurements)))
    power = predict_power(measurements)

    return {"power": power}