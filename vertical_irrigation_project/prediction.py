"""Prediction adapter expected by app.py.

Returns predicted crop water need in millimetres per day (mm/day).
Field area is accepted for interface compatibility but does not change mm/day:
area is used later to convert water depth into litres in optimizer.py.
"""
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).with_name("irrigation_model.joblib")
DEFAULTS = {
    "crop": "Other",
    "soil_moisture_percent": 35.0,
    "temperature_c": 28.0,
    "humidity_percent": 60.0,
    "light_hours": 8.0,
}


def _load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_water_requirement(
    crop,
    area_m2,
    soil_moisture_percent,
    temperature_c=28.0,
    humidity_percent=60.0,
    light_hours=8.0,
):
    """Predict water need in mm/day for one set of field conditions."""
    del area_m2  # mm/day is a depth; area is used by the optimizer for litres.

    numeric_values = {
        "soil_moisture_percent": soil_moisture_percent,
        "temperature_c": temperature_c,
        "humidity_percent": humidity_percent,
        "light_hours": light_hours,
    }
    if not 0 <= float(soil_moisture_percent) <= 100:
        raise ValueError("Soil moisture must be between 0 and 100%.")
    if not 0 <= float(humidity_percent) <= 100:
        raise ValueError("Humidity must be between 0 and 100%.")
    if not -10 <= float(temperature_c) <= 60:
        raise ValueError("Temperature must be between -10°C and 60°C.")
    if not 0 <= float(light_hours) <= 24:
        raise ValueError("Light hours must be between 0 and 24.")

    bundle = _load_bundle()
    features = bundle["features"]
    row = {
        "crop": crop if crop in [
            "Rice", "Wheat", "Maize", "Tomato", "Sugarcane", "Cotton", "Other"
        ] else "Other",
        **numeric_values,
    }
    X = pd.DataFrame([{name: row[name] for name in features}], columns=features)
    predicted = float(bundle["model"].predict(X)[0])
    return max(0.0, predicted)
