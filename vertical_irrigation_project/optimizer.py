"""Irrigation scheduling and water-use optimization.

The optimizer accepts a predicted water requirement from Person 1's model.
All water quantities are estimates; calibrate crop parameters and thresholds
for the local soil, climate, crop stage, and irrigation system before field use.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


# Illustrative crop coefficients (mm/day). Replace with locally validated values.
CROP_DAILY_NEED_MM = {
    "Rice": 6.0,
    "Wheat": 4.0,
    "Maize": 5.0,
    "Tomato": 4.5,
    "Sugarcane": 5.5,
    "Cotton": 4.5,
    "Other": 4.0,
}


@dataclass
class IrrigationPlan:
    crop: str
    next_irrigation: str
    water_liters: float
    duration_minutes: float
    predicted_need_mm: float
    adjusted_need_mm: float
    water_savings_liters: float
    water_savings_percent: float
    reason: str


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def optimize_irrigation(
    crop: str,
    field_area_m2: float,
    soil_moisture_percent: float,
    predicted_need_mm: Optional[float] = None,
    water_limit_liters: float = 5000.0,
    pump_flow_liters_per_minute: float = 20.0,
    irrigation_efficiency_percent: float = 85.0,
    target_soil_moisture_percent: float = 60.0,
    baseline_water_liters: Optional[float] = None,
    preferred_start_hour: int = 6,
    now: Optional[datetime] = None,
    prediction_includes_soil_moisture: bool = False,
) -> IrrigationPlan:
    """Create a practical schedule from predicted water need and field settings.

    Conversion: 1 mm of water over 1 m² = 1 litre.
    Irrigation efficiency accounts for losses: gross water = net water / efficiency.
    """
    if field_area_m2 <= 0:
        raise ValueError("Field area must be greater than zero.")
    if water_limit_liters < 0:
        raise ValueError("Available water limit cannot be negative.")
    if pump_flow_liters_per_minute <= 0:
        raise ValueError("Pump flow rate must be greater than zero.")
    if not 0 <= soil_moisture_percent <= 100:
        raise ValueError("Soil moisture must be between 0 and 100%.")
    if not 1 <= preferred_start_hour <= 23:
        raise ValueError("Preferred start hour must be between 1 and 23.")
    if not 1 <= irrigation_efficiency_percent <= 100:
        raise ValueError("Irrigation efficiency must be between 1 and 100%.")

    crop = crop if crop in CROP_DAILY_NEED_MM else "Other"
    model_need = (
        float(predicted_need_mm)
        if predicted_need_mm is not None
        else CROP_DAILY_NEED_MM[crop]
    )
    if model_need < 0:
        raise ValueError("Predicted water requirement cannot be negative.")

    # The ML model in this project already uses soil moisture as an input.
    # Avoid applying the same moisture reduction twice in that case. For manual
    # values and the illustrative crop fallback, apply the moisture adjustment here.
    if prediction_includes_soil_moisture:
        adjusted_need_mm = model_need
    else:
        moisture_gap = _clamp(
            (target_soil_moisture_percent - soil_moisture_percent)
            / max(target_soil_moisture_percent, 1),
            0.0,
            1.0,
        )
        adjusted_need_mm = model_need * moisture_gap

    efficiency = irrigation_efficiency_percent / 100.0
    requested_liters = (adjusted_need_mm * field_area_m2) / efficiency
    delivered_liters = min(requested_liters, water_limit_liters)
    delivered_liters = max(0.0, delivered_liters)

    duration = delivered_liters / pump_flow_liters_per_minute

    # If no water is needed, do not schedule irrigation.
    now = now or datetime.now()
    if delivered_liters <= 0.01:
        next_irrigation = "No irrigation needed now"
        reason = "Predicted need is zero or soil moisture is at/above the target."
    else:
        start = now.replace(hour=preferred_start_hour, minute=0, second=0, microsecond=0)
        if start <= now:
            start += timedelta(days=1)
        next_irrigation = start.strftime("%a, %d %b %Y at %I:%M %p")
        if requested_liters > water_limit_liters:
            reason = (
                "Water availability limit is lower than the calculated requirement; "
                "the plan is capped at the configured limit."
            )
        else:
            reason = "Schedule targets the next preferred irrigation window and accounts for soil moisture."

    # Compare against a simple crop-default baseline, not against the model's own
    # moisture-aware prediction. This makes the comparison meaningful when the ML
    # model already includes soil moisture as a feature.
    baseline = (
        float(baseline_water_liters)
        if baseline_water_liters is not None
        else CROP_DAILY_NEED_MM[crop] * field_area_m2 / efficiency
    )
    baseline = max(0.0, baseline)
    # Signed difference: positive means less water than baseline; negative means
    # the recommended plan uses more water than the illustrative baseline.
    savings = baseline - delivered_liters
    savings_pct = (savings / baseline * 100.0) if baseline > 0 else 0.0

    return IrrigationPlan(
        crop=crop,
        next_irrigation=next_irrigation,
        water_liters=round(delivered_liters, 2),
        duration_minutes=round(duration, 2),
        predicted_need_mm=round(model_need, 2),
        adjusted_need_mm=round(adjusted_need_mm, 2),
        water_savings_liters=round(savings, 2),
        water_savings_percent=round(savings_pct, 1),
        reason=reason,
    )
