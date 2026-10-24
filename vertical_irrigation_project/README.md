# ML-Based Smart Irrigation Optimizer

A software-only prototype that predicts illustrative daily crop-water demand and passes it to a scheduling optimizer and Streamlit dashboard.

## Project files

- `generate_dataset.py` — generates a **synthetic demonstration dataset**.
- `irrigation_dataset.csv` — created by running the generator.
- `train_model.py` — trains and evaluates a Random Forest regression pipeline.
- `irrigation_model.joblib` — trained model artifact, created during training.
- `model_metrics.json` — held-out evaluation metrics, created during training.
- `prediction.py` — exposes `predict_water_requirement(...)` for the dashboard.
- `optimizer.py` — scheduling, water quantity, duration, limits, and savings calculations.
- `app.py` — dashboard, chart, schedule table, and CSV export.
- `requirements.txt` — Python dependencies.

## Run on Windows

Open a terminal in this project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, you can also run the environment's Python directly after creating it:
`.venv\Scripts\python.exe -m pip install -r requirements.txt`

## Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

The app normally opens in a browser at the local address printed by Streamlit.

## Prediction interface

`prediction.py` exposes this function:

```python
predict_water_requirement(
    crop,
    area_m2,
    soil_moisture_percent,
    temperature_c=28.0,
    humidity_percent=60.0,
    light_hours=8.0,
)
```

It returns a numeric water requirement in **mm/day**. The app calls this function and sends its result to `optimize_irrigation()` in `optimizer.py`.

## Dataset columns

| Column | Meaning |
|---|---|
| `crop` | Crop category |
| `soil_moisture_percent` | Current soil moisture, percent |
| `temperature_c` | Temperature in Celsius |
| `humidity_percent` | Relative humidity, percent |
| `light_hours` | Daily light exposure in hours |
| `water_need_mm` | Target estimate in millimetres per day |

To use real data, replace `irrigation_dataset.csv` with a documented dataset containing these columns, or update the preprocessing code to match your source. Keep the units consistent and document the source, collection method, location, crop stage, and any transformations.

## Research integrity and limitations

**The included dataset is synthetic.** Its target values are generated from a simple illustrative formula so the full software pipeline can run without an external dataset. The Random Forest learns to approximate that formula. Good test scores on this dataset only show that the model reproduces the simulated labels; they do not demonstrate that the model accurately predicts real crop water needs.

For a research-quality evaluation:
1. Replace the synthetic data with a credible agricultural dataset or locally measured data.
2. Confirm that the target represents a meaningful water requirement or irrigation outcome.
3. Use a train/test split that avoids leakage (for time-series data, split by time or growing cycle).
4. Report MAE and RMSE in mm/day, and compare against a simple baseline.
5. Validate crop-specific assumptions with an agronomist or reliable agronomic reference.
6. Report water savings as simulated unless validated against measured water use.

The water-savings comparison uses the optimizer's illustrative fixed crop-default estimate as its baseline. It is not a measured or validated conventional schedule.

This project is a software simulation and does not connect to physical sensors, pumps, or valves.
