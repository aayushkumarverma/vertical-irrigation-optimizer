"""Streamlit dashboard for ML-based irrigation prediction and scheduling."""
from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from optimizer import optimize_irrigation, CROP_DAILY_NEED_MM
from prediction import predict_water_requirement

HERE = Path(__file__).resolve().parent
METRICS_PATH = HERE / "model_metrics.json"

st.set_page_config(page_title="Smart Irrigation Optimizer", page_icon="💧", layout="wide")
st.title("💧 ML-Based Smart Irrigation Optimizer")
st.caption(
    "Predict crop-water demand from field conditions, then generate a water-aware "
    "irrigation schedule. This is a software simulation, not a sensor-connected system."
)

with st.sidebar:
    st.header("Field and environmental conditions")
    crop = st.selectbox("Crop", list(CROP_DAILY_NEED_MM.keys()), index=0)
    area_m2 = st.number_input("Field area (m²)", min_value=1.0, value=500.0, step=50.0)
    soil_moisture = st.slider("Current soil moisture (%)", 0, 100, 35)
    temperature = st.slider("Temperature (°C)", -10, 60, 28)
    humidity = st.slider("Relative humidity (%)", 0, 100, 60)
    light_hours = st.slider("Daily light exposure (hours)", 0.0, 24.0, 8.0, 0.5)
    st.divider()
    st.header("Scheduling constraints")
    water_limit = st.number_input("Available water limit (litres)", min_value=0.0, value=5000.0, step=250.0)
    flow_rate = st.number_input("Pump flow rate (litres/min)", min_value=0.1, value=20.0, step=1.0)
    efficiency = st.slider("Irrigation system efficiency (%)", 1, 100, 85)
    target_moisture = st.slider("Target soil moisture (%)", 10, 100, 60)
    preferred_hour = st.selectbox(
        "Preferred start time", list(range(1, 24)), index=5,
        format_func=lambda h: datetime(2000, 1, 1, h).strftime("%I:00 %p")
    )
    use_custom_prediction = st.checkbox("Enter predicted water need manually", value=False)
    manual_need = st.number_input(
        "Manual predicted water need (mm/day)", min_value=0.0, value=5.0,
        step=0.5, disabled=not use_custom_prediction
    )

prediction_includes_soil_moisture = False
if use_custom_prediction:
    predicted_need = float(manual_need)
    source = "Manual input"
else:
    try:
        predicted_need = predict_water_requirement(
            crop=crop,
            area_m2=area_m2,
            soil_moisture_percent=soil_moisture,
            temperature_c=temperature,
            humidity_percent=humidity,
            light_hours=light_hours,
        )
        source = "Trained ML model"
        prediction_includes_soil_moisture = True
    except (FileNotFoundError, ValueError) as exc:
        st.warning(
            f"ML model is not ready ({exc}). Using the illustrative crop estimate. "
            "Run `python train_model.py` to train the demo model."
        )
        predicted_need = float(CROP_DAILY_NEED_MM.get(crop, CROP_DAILY_NEED_MM["Other"]))
        source = "Illustrative fallback"

try:
    plan = optimize_irrigation(
        crop=crop,
        field_area_m2=area_m2,
        soil_moisture_percent=soil_moisture,
        predicted_need_mm=predicted_need,
        water_limit_liters=water_limit,
        pump_flow_liters_per_minute=flow_rate,
        irrigation_efficiency_percent=efficiency,
        target_soil_moisture_percent=target_moisture,
        preferred_start_hour=preferred_hour,
        prediction_includes_soil_moisture=prediction_includes_soil_moisture,
        baseline_water_liters=(
            CROP_DAILY_NEED_MM.get(crop, CROP_DAILY_NEED_MM["Other"])
            * area_m2 / (efficiency / 100)
        ),
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Current conditions")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Crop", crop)
c2.metric("Field area", f"{area_m2:,.0f} m²")
c3.metric("Soil moisture", f"{soil_moisture}%")
c4.metric("Prediction source", source)

st.subheader("Recommended irrigation plan")
a, b, c = st.columns(3)
a.metric("Predicted water need", f"{predicted_need:.2f} mm/day")
b.metric("Water to supply", f"{plan.water_liters:,.0f} L")
c.metric("Estimated duration", f"{plan.duration_minutes:,.1f} min")
difference_label = (
    "Estimated water saved"
    if plan.water_savings_liters >= 0
    else "Estimated additional water vs. baseline"
)
st.caption(
    f"{difference_label}: {abs(plan.water_savings_liters):,.0f} L "
    f"({abs(plan.water_savings_percent):.1f}% "
    f"{'less' if plan.water_savings_liters >= 0 else 'more'} than the defined baseline). "
    "This is a simulation estimate, not a measured field result."
)

st.info(f"**Recommended time:** {plan.next_irrigation}\n\n**Reason:** {plan.reason}")

schedule = pd.DataFrame([{
    "Crop": plan.crop,
    "Recommended time": plan.next_irrigation,
    "Predicted need (mm/day)": plan.predicted_need_mm,
    "Moisture-adjusted need (mm)": plan.adjusted_need_mm,
    "Water quantity (L)": plan.water_liters,
    "Duration (minutes)": plan.duration_minutes,
    "Water-use difference vs baseline (L)": plan.water_savings_liters,
}])
st.subheader("Schedule table")
st.dataframe(schedule, use_container_width=True, hide_index=True)

st.subheader("Water quantity comparison")
baseline_liters = (
    CROP_DAILY_NEED_MM.get(crop, CROP_DAILY_NEED_MM["Other"])
    * area_m2 / (efficiency / 100)
)
comparison = pd.DataFrame({
    "Scenario": ["Baseline estimate", "Optimized schedule"],
    "Water (litres)": [round(baseline_liters, 2), plan.water_liters],
})
st.bar_chart(comparison.set_index("Scenario"))

csv = schedule.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download irrigation schedule (CSV)",
    data=csv,
    file_name="irrigation_schedule.csv",
    mime="text/csv",
)

if METRICS_PATH.exists():
    with st.expander("Model evaluation (training report)"):
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write("These metrics describe performance on the held-out portion of the training dataset.")
        st.json(metrics)
        st.warning(
            "If the dataset is synthetic, these scores do not establish real-world agronomic accuracy."
        )

with st.expander("Assumptions and limitations"):
    st.markdown("""
    - The model output is interpreted as **millimetres of water per day (mm/day)**.
    - Conversion: **1 mm of water over 1 m² = 1 litre**.
    - Irrigation efficiency adjusts for expected delivery losses.
    - Water supplied is capped by the available-water limit.
    - The baseline difference can be positive (less water) or negative (more water).
    - Duration = water quantity ÷ pump flow rate (litres/minute).
    - The included dataset is synthetic demonstration data unless replaced with real, documented data.
    - Crop defaults and simulated labels are not agronomic advice.
    - Validate model predictions, soil thresholds, and the baseline with local agronomic evidence
      before making real irrigation decisions.
    """)
