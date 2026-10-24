"""Generate a clearly labelled synthetic dataset for the irrigation ML demo.

IMPORTANT:
This is simulated data for demonstrating the software pipeline. It is NOT real
field data and must not be reported as measured agricultural evidence.
Replace it with a properly sourced/local dataset before drawing research conclusions.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT = Path(__file__).with_name("irrigation_dataset.csv")
RANDOM_SEED = 42

CROP_BASE_NEED_MM = {
    "Rice": 6.0,
    "Wheat": 4.0,
    "Maize": 5.0,
    "Tomato": 4.5,
    "Sugarcane": 5.5,
    "Cotton": 4.5,
    "Other": 4.0,
}


def estimate_demo_water_need(crop, soil_moisture, temperature, humidity, light_hours):
    """Create a plausible illustrative target, not an agronomic recommendation."""
    base = CROP_BASE_NEED_MM[crop]

    # Warm weather and more sunlight increase illustrative demand.
    temperature_factor = np.clip(1.0 + (temperature - 25.0) * 0.018, 0.65, 1.40)
    light_factor = np.clip(0.80 + light_hours * 0.045, 0.80, 1.35)

    # Lower relative humidity increases illustrative demand.
    humidity_factor = np.clip(1.20 - humidity * 0.004, 0.75, 1.15)

    # Wet soil reduces the immediate need; dry soil increases it.
    moisture_factor = np.clip(1.35 - soil_moisture * 0.009, 0.40, 1.35)

    return max(0.0, base * temperature_factor * light_factor *
               humidity_factor * moisture_factor)


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    crop_names = list(CROP_BASE_NEED_MM)

    for _ in range(3000):
        crop = rng.choice(crop_names)
        soil_moisture = float(rng.uniform(10, 85))
        temperature = float(rng.uniform(12, 42))
        humidity = float(rng.uniform(20, 95))
        light_hours = float(rng.uniform(0, 14))

        calculated = estimate_demo_water_need(
            crop, soil_moisture, temperature, humidity, light_hours
        )
        # Small noise avoids a perfectly deterministic toy dataset.
        target = max(0.0, calculated + float(rng.normal(0, 0.12)))

        rows.append({
            "crop": crop,
            "soil_moisture_percent": round(soil_moisture, 2),
            "temperature_c": round(temperature, 2),
            "humidity_percent": round(humidity, 2),
            "light_hours": round(light_hours, 2),
            "water_need_mm": round(target, 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)
    print(f"Created {len(df)} synthetic demo rows: {OUTPUT}")
    print("Reminder: this is simulated data, not real field measurements.")


if __name__ == "__main__":
    main()
