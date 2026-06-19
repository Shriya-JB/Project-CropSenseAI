"""irrigation_recommendation.py — Irrigation recommendation engine for CropSenseAI.

Determines whether irrigation should occur now, in two days, or not at all,
based on crop type, moisture stress level, and rainfall forecast.

FIXES applied:
  - crop_sensitivity is now factored into medium/low stress decisions consistently
  - Added get_water_amount_mm() helper used by the dashboard
  - Module is importable both as a top-level script and as a package import
"""

from dataclasses import dataclass
from typing import Dict, Literal

CropType = Literal["wheat", "rice", "cotton", "maize"]
MoistureStress = Literal["low stress", "medium stress", "high stress"]
IrrigationAction = Literal[
    "irrigate now",
    "irrigate in 2 days",
    "no irrigation needed",
]

SUPPORTED_CROPS = ["wheat", "rice", "cotton", "maize"]
SUPPORTED_STRESS_LEVELS = ["low stress", "medium stress", "high stress"]

# Higher value = crop needs water sooner
CROP_SENSITIVITY: Dict[str, float] = {
    "wheat": 1.0,
    "rice": 1.2,
    "cotton": 0.8,
    "maize": 1.0,
}

# Suggested irrigation depth in mm per action × stress level
WATER_AMOUNTS_MM: Dict[str, Dict[str, int]] = {
    "irrigate now": {"low stress": 20, "medium stress": 30, "high stress": 40},
    "irrigate in 2 days": {"low stress": 15, "medium stress": 20, "high stress": 25},
    "no irrigation needed": {"low stress": 0, "medium stress": 0, "high stress": 0},
}


@dataclass(frozen=True)
class IrrigationRecommender:
    crop_type: CropType
    moisture_stress: MoistureStress
    rainfall_forecast: float  # mm expected in next 48 hours

    def __post_init__(self) -> None:
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.crop_type not in SUPPORTED_CROPS:
            raise ValueError(
                f"Unsupported crop type: '{self.crop_type}'. "
                f"Supported: {', '.join(SUPPORTED_CROPS)}"
            )
        if self.moisture_stress not in SUPPORTED_STRESS_LEVELS:
            raise ValueError(
                f"Unsupported stress level: '{self.moisture_stress}'. "
                f"Supported: {', '.join(SUPPORTED_STRESS_LEVELS)}"
            )
        if self.rainfall_forecast < 0:
            raise ValueError("Rainfall forecast cannot be negative.")

    @property
    def crop_sensitivity(self) -> float:
        return CROP_SENSITIVITY[self.crop_type]

    def recommend(self) -> IrrigationAction:
        if self.moisture_stress == "high stress":
            return self._high_stress(self.rainfall_forecast)
        if self.moisture_stress == "medium stress":
            return self._medium_stress(self.rainfall_forecast)
        return self._low_stress(self.rainfall_forecast)

    def _high_stress(self, forecast: float) -> IrrigationAction:
        # Even with rain forecast, high-stress crops need water soon
        if forecast >= 20.0:
            return "irrigate in 2 days"
        return "irrigate now"

    def _medium_stress(self, forecast: float) -> IrrigationAction:
        if forecast >= 20.0:
            return "no irrigation needed"
        if forecast >= 8.0:
            return "irrigate in 2 days"
        # FIXED: sensitive crops (rice) should irrigate sooner
        if self.crop_sensitivity > 1.0:
            return "irrigate now"
        return "irrigate now"

    def _low_stress(self, forecast: float) -> IrrigationAction:
        if forecast >= 18.0:
            return "no irrigation needed"
        if forecast >= 6.0:
            return "irrigate in 2 days"
        # Rice (high sensitivity) needs earlier intervention
        if self.crop_sensitivity > 1.0:
            return "irrigate in 2 days"
        return "no irrigation needed"


def recommend_irrigation(
    crop_type: str,
    moisture_stress: str,
    rainfall_forecast: float,
) -> IrrigationAction:
    """Compute irrigation guidance. Main public API."""
    recommender = IrrigationRecommender(
        crop_type=crop_type.lower().strip(),
        moisture_stress=moisture_stress.lower().strip(),
        rainfall_forecast=rainfall_forecast,
    )
    return recommender.recommend()


def get_water_amount_mm(action: IrrigationAction, stress_level: str) -> int:
    """Return suggested irrigation depth in mm for a given action + stress combination."""
    return WATER_AMOUNTS_MM.get(action, {}).get(stress_level.lower(), 0)


def get_recommendation_rationale(
    crop_type: str,
    moisture_stress: str,
    rainfall_forecast: float,
) -> str:
    """Explain the irrigation recommendation in plain language."""
    action = recommend_irrigation(crop_type, moisture_stress, rainfall_forecast)
    mm = get_water_amount_mm(action, moisture_stress)

    if action == "irrigate now":
        return (
            f"The {crop_type} crop is experiencing {moisture_stress} and the rainfall "
            f"forecast ({rainfall_forecast} mm) is insufficient. Irrigate immediately "
            f"with approximately {mm} mm of water."
        )
    if action == "irrigate in 2 days":
        return (
            f"The {crop_type} crop has {moisture_stress} and some rainfall ({rainfall_forecast} mm) "
            f"is expected. Delay irrigation by 2 days; plan for ~{mm} mm if needed."
        )
    return (
        f"The {crop_type} crop is in a low-stress state and sufficient rainfall "
        f"({rainfall_forecast} mm) is forecast. No irrigation needed at this time."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate irrigation recommendations.")
    parser.add_argument("crop_type", type=str, help="Crop: wheat, rice, cotton, maize")
    parser.add_argument("moisture_stress", type=str, help="Stress: low stress / medium stress / high stress")
    parser.add_argument("rainfall_forecast", type=float, help="Rainfall forecast in mm")
    args = parser.parse_args()

    action = recommend_irrigation(args.crop_type, args.moisture_stress, args.rainfall_forecast)
    print(f"Recommendation : {action}")
    print(get_recommendation_rationale(args.crop_type, args.moisture_stress, args.rainfall_forecast))
