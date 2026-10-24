"""Train and evaluate the irrigation water-need regression model."""
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "irrigation_dataset.csv"
MODEL_PATH = HERE / "irrigation_model.joblib"
METRICS_PATH = HERE / "model_metrics.json"

CATEGORICAL_FEATURES = ["crop"]
NUMERIC_FEATURES = [
    "soil_moisture_percent",
    "temperature_c",
    "humidity_percent",
    "light_hours",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "water_need_mm"


def main():
    if not DATA_PATH.exists():
        # Generate demo data so a first run is straightforward.
        from generate_dataset import main as generate_demo_dataset
        generate_demo_dataset()

    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=FEATURES + [TARGET])
    if len(df) < 30:
        raise ValueError("Please provide at least 30 complete rows to train the model.")
    if (df[TARGET] < 0).any():
        raise ValueError("water_need_mm must not contain negative values.")

    X = df[FEATURES]
    y = df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("crop", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae_mm_per_day": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse_mm_per_day": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "dataset_note": (
            "The included generated CSV is synthetic demo data. If you replace it with real data, "
            "document the source and interpretation of these metrics in the project report."
        ),
        "features": FEATURES,
        "target": TARGET,
    }

    joblib.dump({
        "model": model,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
    }, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Model training complete.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(json.dumps(metrics, indent=2))
    print("\nImportant: metrics on synthetic data measure reproduction of the demo formula,")
    print("not real-world irrigation accuracy.")


if __name__ == "__main__":
    main()
