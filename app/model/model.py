from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

model_name = "xgboost_regression" + __version__ + ".pkl"

with open(BASE_DIR / model_name, "rb") as f:
    model = pickle.load(f)

def predict_power(measurements:np.array) -> float:
    """Predict the active power using input sensor 
    measurements.

    Args:
        measurements (np.array): Array of shape 7.

    Returns:
        float: Active power in W
    """
    prediction = model.predict(measurements)

    return prediction[0]